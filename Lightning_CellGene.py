from lightning import LightningModule, Trainer, seed_everything
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.plugins.environments import SLURMEnvironment
from mmvae.data.pipes import CellCensusPipeLine
import torchdata.dataloader2 as dl

class VAE(pl.LightningModule):
    """
    Standard VAE architecture without adversarial feedback, discriminator networks, or multiple modalities
    """
    def __init__(self, beta: float, batch_size: int) -> None:
        super().__init__()
        self.beta = beta
        self.batch_size = batch_size
        self.encoder = nn.Sequential(
            nn.Linear(60664, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 60664),
            nn.BatchNorm1d(60664),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.mean = nn.Linear(128, 64)
        self.var = nn.Linear(128, 64)
        self.save_hyperparameters()

    def _reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(var)
        return mean + var*eps
    
    def forward(self, x: torch.Tensor, target: torch.Tensor = None) -> tuple[torch.Tensor]:
        x = self.encoder(x)
        mean = self.mean(x)
        var = self.var(x)
        z = self._reparameterize(mean, torch.exp(0.5 * var))
        x_hat = self.decoder(z)
        return x_hat, mean, var
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> tuple[torch.Tensor]:
        inputs = batch[0]
        output, mean, logvar = self(inputs)
        recon_loss = F.mse_loss(output, inputs.to_dense(), reduction='sum')
        KLDiv_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * KLDiv_loss
        self.log_dict({"train/loss": loss.item() / inputs.numel(), "train/KL_loss": KLDiv_loss.item() / mean.numel(), "train/Recon_Loss": recon_loss.item() / output.numel()},
                       on_epoch=True, on_step=True, logger=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> tuple[torch.Tensor]:
        inputs = batch[0]
        output, mean, logvar = self(inputs)
        recon_loss = F.mse_loss(output, inputs.to_dense(), reduction='mean')
        KLDiv_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * KLDiv_loss
        self.log_dict({"val/loss": loss, "val/KL_loss": KLDiv_loss, "val/Recon_Loss": recon_loss},
                       on_epoch=True, on_step=True, logger=True, batch_size=self.batch_size)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> tuple[torch.Tensor]:
        inputs = batch[0]
        output, mean, logvar = self(inputs)
        recon_loss = F.mse_loss(output, inputs.to_dense(), reduction='mean')
        KLDiv_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * KLDiv_loss
        self.log_dict({"test/loss": loss, "test/KL_loss": KLDiv_loss, "test/Recon_Loss": recon_loss},
                       on_epoch=True, on_step=True, logger=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    
class WorkerShutdownCallback(pl.Callback):
    """Custom Lightning Callback that ensures that all Dataloader worker processes are
        shut off if there is an error or once training completes.
    """
    def __init__(self, data_laoders: list[dl.DataLoader2]) -> None:
        super().__init__()
        self.data_loaders = data_laoders
    
    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        for dl in self.data_loaders:
            dl.shutdown()
        return super().on_exception(trainer, pl_module, exception)
    
    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # Called when fit, validate, test, predict, or tune ends.
        for dl in self.data_loaders:
            dl.shutdown()
        return super().teardown(trainer, pl_module, stage)

def configure_dataloaders(seed: int, batch_size: int, num_workers: int):
    # Convienence wrapper for dataloader config
    pipeline = CellCensusPipeLine(directory_path="/active/debruinz_project/CellCensus_3M/", masks=["3m_human_chunk*.npz"], batch_size=batch_size)
    train_pipe, val_pipe = pipeline.random_split(weights={"train": 0.8, "test": 0.2}, total_length=3000000, seed=seed)
    train_loader = dl.DataLoader2(datapipe=train_pipe, datapipe_adapter_fn=None, reading_service=dl.MultiProcessingReadingService(num_workers=num_workers))
    val_loader = dl.DataLoader2(datapipe=val_pipe, datapipe_adapter_fn=None, reading_service=dl.MultiProcessingReadingService(num_workers=num_workers))
    return train_loader, val_loader

def main(seed, batch_size):
    seed_everything(seed, workers=True)
    train_loader, val_loader = configure_dataloaders(seed, batch_size, num_workers = 2)
    model = VAE(0.15, batch_size)
    logger = TensorBoardLogger(save_dir="/active/debruinz_project/tensorboard_logs/tony_boos/", version=0.1, name='Baseline_3M', default_hp_metric=False)
    logger.log_hyperparams(model.hparams)
    callbacks = [WorkerShutdownCallback([train_loader, val_loader]),
                 EarlyStopping("val/Recon_Loss_epoch", stopping_threshold=0.1)]
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=logger, max_epochs=30, enable_progress_bar=False, enable_checkpointing=True, deterministic=True,
                         plugins=[SLURMEnvironment(auto_requeue=False)], callbacks=callbacks)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    #trainer.test(model)

if __name__ == "__main__":
    main(42, 256)