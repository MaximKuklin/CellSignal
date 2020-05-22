from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import pytorch_lightning as pl


checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=True,
    verbose=True,
    monitor='val_loss',
    mode='min',
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=5,
    verbose=True,
    mode='min'
)


class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)


def get_checkpoint_callback(exp: int, period=1, save_every_n=False):
    save_top_k = True

    if save_every_n:
        save_top_k = -1

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd()+f'/checkpoints/exp_{str(exp)}/',
        save_top_k=save_top_k,
        verbose=True,
        monitor='val_loss',
        mode='min',
        period=period,
    )
    return checkpoint_callback
