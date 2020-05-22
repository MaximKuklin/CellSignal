from argparse import ArgumentParser
import pytorch_lightning as pl
from engine.train import ClassificationModel
from pytorch_lightning.loggers import TensorBoardLogger
from callbacks.callbacks import checkpoint_callback, early_stop_callback, get_checkpoint_callback, ModelCheckpointAtEpochEnd


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp', type=int, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--model_name', type=str, default='resnet18')
    # parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--arcface_loss', action='store_true')
    parser.add_argument('--arcmargin', action='store_true')
    parser.add_argument('--focal_loss', action='store_true')
    parser.add_argument('--initial_lr', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=10000)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    logger = TensorBoardLogger('checkpoints/exp_'+str(args.exp), name='logs', version='')
    all_loggers = TensorBoardLogger("tb_logs", name="trainings")
    logger.log_hyperparams(args)

    model = ClassificationModel(hparams=args)

    save_every_n = get_checkpoint_callback(args.exp, period=3, save_every_n=True)
    # save_best = get_checkpoint_callback(args.exp, period=1, save_every_n=False)

    trainer = pl.Trainer(
        default_root_dir='checkpoints/'+str(args.exp),
        checkpoint_callback=save_every_n,
        # early_stop_callback=early_stop_callback,
        max_epochs=args.max_epoch, gpus=[0], logger=[logger, all_loggers],
    )

    trainer.fit(model)
    trainer.save_checkpoint('exp')
