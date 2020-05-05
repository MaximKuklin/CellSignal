from argparse import ArgumentParser
import pytorch_lightning as pl
from engine.train import ClassificationModel
from pytorch_lightning.loggers import TensorBoardLogger
from callbacks.callbacks import checkpoint_callback, early_stop_callback


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--model_name', type=str, default='resnet18')
    # parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--initial_lr', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=10)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    logger = TensorBoardLogger("tb_logs", name="test")
    logger.log_hyperparams(args)

    model = ClassificationModel(hparams=args)
    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        max_epochs=args.max_epoch, gpus=[0], logger=logger
    )

    trainer.fit(model)
    trainer.save_checkpoint('exp')
