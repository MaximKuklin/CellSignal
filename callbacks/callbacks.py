from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os

checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=True,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix='checkpoints/',
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=5,
    verbose=True,
    mode='min'
)