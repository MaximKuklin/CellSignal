import torch
from torch.utils.data import DataLoader
from torchvision import models
import pytorch_lightning as pl
from models import resnet
import albumentations as A
from engine.dataloader import CellDataset
from albumentations.pytorch import ToTensorV2


class ClassificationModel(pl.LightningModule):
    def __init__(self, hparams):
        super(ClassificationModel, self).__init__()
        self.hparams = hparams
        self.model = resnet.get_resnet(self.hparams.model_name)  # TODO: make model choice
        self.input_size = self.hparams.input_size
        self.num_workers = self.hparams.num_workers
        self.batch_size = self.hparams.batch_size
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        transforms_train = A.Compose([A.Resize(300, 300), A.HorizontalFlip(p=0.5),
                                      A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
                                      A.ToFloat(max_value=255), ToTensorV2()])
        transforms_val = A.Compose([A.Resize(300, 300), A.ToFloat(max_value=255), ToTensorV2()])

        self.train_set = CellDataset(self.hparams.root, 'train', transforms=transforms_train)
        self.val_set = CellDataset(self.hparams.root, 'val', transforms=transforms_val)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_set, batch_size=self.hparams.batch_size,
                                num_workers=self.hparams.num_workers, shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_set, batch_size=self.hparams.batch_size,
                                num_workers=self.hparams.num_workers)
        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.initial_lr)
        # scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5, last_epoch=-1)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, targets = batch
        output = self.forward(images)
        loss = self.loss_fn(output, targets)
        pred = output.argmax(dim=1)
        correct = (targets==pred).sum().item()/float(len(targets))
        progress_bar = {'accuracy': correct}
        return {'loss': loss, 'accuracy': correct, 'progress_bar':progress_bar}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = torch.tensor([x['accuracy'] for x in outputs], dtype=float).mean()
        logs = {'avg_loss': avg_loss, 'avg_accuracy': avg_accuracy}
        tensorboard_logs = {'train/avg_loss': avg_loss, 'train/avg_accuracy': avg_accuracy}
        results = {'progress_bar': logs, 'log': tensorboard_logs}
        return results

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        output = self.forward(images)
        loss = self.loss_fn(output, targets)
        pred = output.argmax(dim=1)
        correct = (targets==pred).sum().item()/float(len(targets))
        return {'val_loss': loss, 'val_accuracy': correct}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.tensor([x['val_accuracy'] for x in outputs], dtype=float).mean()

        logs = {'avg_val_loss': avg_loss, 'avg_val_accuracy': avg_accuracy}
        tensorboard_logs = {'val/avg_val_loss': avg_loss, 'val/avg_val_accuracy': avg_accuracy}

        results = {'val_loss': avg_loss, 'progress_bar': logs, 'log': tensorboard_logs}
        return results

