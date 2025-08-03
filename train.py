import os

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         RichModelSummary, RichProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import SimpleProfiler
import pickle


import time
from pytorch_lightning import Callback

class EpochTimeLogger(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_time = 0.0
        self._train_batch_start = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._train_batch_start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._train_batch_start is not None:
            self.train_epoch_time += time.time() - self._train_batch_start

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log("train_epoch_duration", self.train_epoch_time, prog_bar=True, on_epoch=True, logger=True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    torch.use_deterministic_algorithms(True)
    torch.multiprocessing.set_start_method("spawn")
    pl.seed_everything(conf.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    output_dir = HydraConfig.get().runtime.output_dir

    # get model name for profiler
    model_name = getattr(conf.model, "name", None)
    if model_name is None:
        model_name = conf.model.target._target_.split(".")[-1]
        
    if conf.wandb != "disable":
        logger = WandbLogger(
            project="EMP",
            name=conf.output,
            mode=conf.wandb,
            log_model="all",
            resume=conf.checkpoint is not None,
        )
    else:
        logger = TensorBoardLogger(save_dir=output_dir, name=f'logs/{model_name}')

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}_emp_small",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
            every_n_train_steps=100,
        ),
        RichModelSummary(max_depth=1),
        # RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
        EpochTimeLogger(),
    ]

    profiler = SimpleProfiler(dirpath=".", filename=f'profiler_{model_name}.txt')
    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="auto",
        devices=1,
        strategy="auto",
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
        profiler=profiler
    )

    model = instantiate(conf.model.target)
    datamodule = instantiate(conf.datamodule)

    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)

    # save the profiler results
    with open(f"profiler_{model_name}.pkl", "wb") as f:
        pickle.dump(profiler, f)
    print(f"Profiler pickled and saved to profiler_{model_name}.pkl")


if __name__ == "__main__":
    main()