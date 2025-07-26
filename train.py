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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    torch.use_deterministic_algorithms(True)
    torch.multiprocessing.set_start_method("spawn")
    pl.seed_everything(conf.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    output_dir = HydraConfig.get().runtime.output_dir

    if conf.wandb != "disable":
        logger = WandbLogger(
            project="EMP",
            name=conf.output,
            mode=conf.wandb,
            log_model="all",
            resume=conf.checkpoint is not None,
        )
    else:
        logger = TensorBoardLogger(save_dir=output_dir, name="logs")

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
    ]

    profiler = SimpleProfiler(dirpath=".", filename="profiler.txt")
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
    with open("profiler.pkl", "wb") as f:
        pickle.dump(profiler, f)
    print("Profiler pickled and saved to profiler.pkl")


if __name__ == "__main__":
    main()