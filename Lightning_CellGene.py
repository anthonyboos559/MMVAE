import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from mmvae.data.pipes import CellCensusPipeLine
import torchdata.dataloader2 as dl

class VAE(pl.LightningModule):
    """
    Standard VAE architecture without adversarial feedback, discriminator networks, or multiple modalities
    """
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(60664, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 60664),
            nn.BatchNorm1d(60664),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1)
        )
        self.mean = nn.Linear(64, 32)
        self.var = nn.Linear(64, 32)
        self.beta = beta
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
        self.log_dict({"train/loss": loss.item() / inputs.numel(), "train/KL loss": KLDiv_loss.item() / mean.numel(), "train/Recon Loss": recon_loss.item() / output.numel()},
                       on_epoch=True, on_step=True, logger=True, batch_size=32)
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> tuple[torch.Tensor]:
        inputs = batch[0]
        output, mean, logvar = self(inputs)
        recon_loss = F.mse_loss(output, inputs.to_dense(), reduction='sum')
        KLDiv_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * KLDiv_loss
        self.log_dict({"train/loss": loss.item() / inputs.numel(), "train/KL loss": KLDiv_loss.item() / mean.numel(), "train/Recon Loss": recon_loss.item() / output.numel()},
                       on_epoch=True, on_step=True, logger=True, batch_size=32)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> tuple[torch.Tensor]:
        inputs, _ = batch
        output, mean, logvar = self(inputs)
        recon_loss = F.mse_loss(output, inputs.to_dense(), reduction='sum')
        KLDiv_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * KLDiv_loss
        self.log_dict({"train/loss": loss.item() / inputs.numel(), "train/KL loss": KLDiv_loss.item() / mean.numel(), "train/Recon Loss": recon_loss.item() / output.numel()},
                       on_epoch=True, on_step=True, logger=True, batch_size=32)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

def main():
    pipeline = CellCensusPipeLine(directory_path="/active/debruinz_project/CellCensus_3M/", masks=["3m_human_chunk*.npz"], batch_size=32)
    train_pipe, val_pipe = pipeline.random_split(weights={"train": 0.8, "test": 0.2}, total_length=3000000, seed=42)
    train_loader = dl.DataLoader2(datapipe=train_pipe, datapipe_adapter_fn=None, reading_service=dl.MultiProcessingReadingService(num_workers=0))
    val_loader = dl.DataLoader2(datapipe=val_pipe, datapipe_adapter_fn=None, reading_service=dl.MultiProcessingReadingService(num_workers=0))
    # train_loader = ChunkedCellCensusDataLoader(None, directory_path="/active/debruinz_project/human_data/python_data",
    #                                             masks=['human_chunk_[1-7][0-9]*', 'human_chunk_[1-9][.]*'], batch_size=32, num_workers=0)
    # val_loader = ChunkedCellCensusDataLoader(None, directory_path="/active/debruinz_project/human_data/python_data",
    #                                             masks=['human_chunk_[89][0-9]*'], batch_size=32, num_workers=0)
    model = VAE(0.15)
    logger = TensorBoardLogger(save_dir="/active/debruinz_project/tony_boos/tb_logs", version=0.2, name='New_Arch_Test', default_hp_metric=False)
    logger.log_hyperparams(model.hparams)
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=logger, max_epochs=5, enable_progress_bar=False, enable_checkpointing=False)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    #trainer.test(model)


if __name__ == "__main__":
    main()