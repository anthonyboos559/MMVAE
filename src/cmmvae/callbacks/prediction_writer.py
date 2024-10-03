from typing import Any, Optional, Sequence
import os
import warnings
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
import pandas as pd
import numpy as np
import h5py
import torch

from cmmvae.constants import REGISTRY_KEYS as RK


def save_to_hdf5(
    data: np.ndarray,
    metadata: pd.DataFrame,
    hdf5_filepath: str,
    key: str,
    strict: bool = True,
):
    """
    Save numpy array `data` and pandas DataFrame `metadata` to HDF5 file.
    Each column in the metadata DataFrame will be stored as a separate dataset.
    """

    with h5py.File(hdf5_filepath, "a") as h5file:
        group = h5file[key]

        if RK.PREDICT_SAMPLES in group and RK.METADATA in group:
            sample_ds = group[RK.PREDICT_SAMPLES]
            metadata_ds = group[RK.METADATA]

            new_size = sample_ds.shape[0] + data.shape[0]
            sample_ds.resize(new_size, axis=0)
            sample_ds[-data.shape[0] :] = data  # Append new batch

            # Append metadata column-wise
            for col in metadata.columns:
                column_data = metadata[col].values
                if col not in metadata_ds:
                    if strict:
                        raise RuntimeError(
                            f"metadata column {col} not in h5file for group_key {key}/{RK.METADATA}/{col}"
                        )
                    else:
                        continue

                metadata_ds[col].resize(new_size, axis=0)
                metadata_ds[col][-len(column_data) :] = column_data
        else:
            # Create a new dataset for `data`
            group.create_dataset(
                RK.PREDICT_SAMPLES,
                data=data,
                maxshape=(None,) + data.shape[1:],
                chunks=True,
            )

            # Create metadata datasets, one for each column
            metadata_group = group.create_group(RK.METADATA)
            for col in metadata.columns:
                metadata_group.create_dataset(
                    col, data=metadata[col].values, maxshape=(None,), chunks=True
                )


def load_from_hdf5(hdf5_filepath: str, key: str):
    """
    Load numpy array `data` and pandas DataFrame `metadata` from HDF5 file.
    """
    data = None
    metadata = None
    embedding = None

    with h5py.File(hdf5_filepath, "r") as h5file:
        group = h5file[key]

        if RK.PREDICT_SAMPLES in group:
            data = group[RK.PREDICT_SAMPLES][:]

        if RK.METADATA in group:
            # Load metadata as individual columns
            metadata_group = group[RK.METADATA]
            metadata = pd.DataFrame(
                {col: metadata_group[col][:] for col in metadata_group.keys()}
            )

        if RK.UMAP_EMBEDDINGS in group:
            embedding = group[RK.UMAP_EMBEDDINGS][:]

    return data, metadata, embedding


class PredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        root_dir: str,
        experiment_name: str = "",
        run_name: str = "",
        hdf5_filename: str = "predictions.h5",
    ):
        super().__init__(write_interval="batch")
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.hdf5_filename = hdf5_filename
        self._curr_size = 0  # Keeps track of total rows written so far

        if os.path.exists(self.hdf5_filepath):
            warnings.warn(
                f"PredictionWriter initialized with hdf5_filepath that already exists: {self.hdf5_filepath}"
            )

    @property
    def hdf5_filepath(self):
        return os.path.join(self.save_dir, self.hdf5_filename)

    @property
    def save_dir(self):
        return os.path.join(self.root_dir, self.experiment_name, self.run_name)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        if not isinstance(prediction, dict) or not all(
            isinstance(p, (tuple, list))
            and len(p) == 2
            and (
                isinstance(p[0], (torch.Tensor, np.ndarray))
                and isinstance(p[1], pd.DataFrame)
            )
            for p in prediction.values()
        ):
            raise ValueError(
                f"Prediction must be a dictionary of type 'dict[str, tuple[torch.Tensor, pd.DataFrame]]' (got {type(prediction)})"
            )

        for key, (data, metadata) in prediction.items():
            data = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
            save_to_hdf5(data, metadata, self.hdf5_filepath, key)

        self._curr_size += list(prediction.values())[0][0].shape[
            0
        ]  # Increment by batch size

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        super().on_predict_start(trainer, pl_module)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._curr_size = 0  # Reset after the epoch ends
        super().on_predict_epoch_end(trainer, pl_module)