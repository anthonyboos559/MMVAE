import torch
import torch.nn as nn
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser

class VAE(pl.LightningModule):
    """
    Standard VAE architecture without adversarial feedback, discriminator networks, or multiple modalities
    """
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(60664, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 60664),
            nn.ReLU()
        )
        self.mean = nn.Linear(128, 32)
        self.var = nn.Linear(128, 32)
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
        inputs, target = batch
        output, mean, logvar = self(inputs)
        recon_loss = torch.mean(torch.nn.MSELoss(reduction='sum')(output, inputs.to_dense()))
        KLDiv_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()))
        loss = recon_loss + self.beta * KLDiv_loss
        self.log_dict({"train/loss": loss, "train/KL loss": KLDiv_loss, "train/Recon Loss": recon_loss}, on_epoch=True, on_step=True, logger=True)
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> tuple[torch.Tensor]:
        inputs, target = batch
        output, mean, logvar = self(inputs)
        recon_loss = torch.mean(torch.nn.MSELoss(reduction='sum')(output, inputs.to_dense()))
        KLDiv_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()))
        loss = recon_loss + self.beta * KLDiv_loss
        self.log_dict({"valid/loss": loss, "valid/KL loss": KLDiv_loss, "valid/Recon Loss": recon_loss}, on_epoch=True, on_step=True, logger=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> tuple[torch.Tensor]:
        inputs, target = batch
        output, mean, logvar = self(inputs)
        recon_loss = torch.mean(torch.nn.MSELoss(reduction='sum')(output, inputs.to_dense()))
        KLDiv_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()))
        loss = recon_loss + self.beta * KLDiv_loss
        self.log_dict({"test/loss": loss, "test/KL loss": KLDiv_loss, "test/Recon Loss": recon_loss}, on_epoch=True, on_step=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

import mmvae.data.utils as utils
def configure_singlechunk_dataloaders(
    data_file_path: str,
    metadata_file_path: str,
    train_ratio: float,
    batch_size: int,
    device: torch.device,
    test_batch_size: int = None
):
    """
    Splits a csr_matrix provided by data_file_path with equal length metadata_file_path by train_ratio 
        which is a floating point between 0-1 (propertion of training data to test data). 
    
    If device is not None -> the entire dataset will be loaded on device at once.
    """
    if not test_batch_size:
        test_batch_size = batch_size
        
    (train_data, train_metadata), (validation_data, validation_metadata) = utils.split_data_and_metadata(
        data_file_path,
        metadata_file_path,
        train_ratio)
    
    from mmvae.data.datasets.CellCensusDataSet import CellCensusDataset, collate_fn
    if device:
        train_data = train_data.to(device)
        validation_data = validation_data.to(device)
        
    train_dataset = CellCensusDataset(train_data, train_metadata)
    test_dataset = CellCensusDataset(validation_data, validation_metadata)
    
    from torch.utils.data import DataLoader
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=test_batch_size,
            collate_fn=collate_fn,
        )
    )

def main():
    model = VAE(1)
    logger = TensorBoardLogger(save_dir="/active/debruinz_project/tony_boos/tb_logs", version=0, name='Lightning Test')
    logger.log_hyperparams(model.hparams)
    train_loader, val_loader = configure_singlechunk_dataloaders(
        data_file_path="/active/debruinz_project/CellCensus_3M/3m_human_chunk_10.npz",
        metadata_file_path="/active/debruinz_project/CellCensus_3M/3m_human_metadata_10.csv",
        train_ratio=0.8,
        batch_size=32,
        device="cuda"
    )
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=logger, max_epochs=30)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    #trainer.test(model)


if __name__ == "__main__":
    main()