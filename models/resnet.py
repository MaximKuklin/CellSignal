import torch
import torchvision.models as models


def get_resnet(name, num_classes=1108, pretrained=True):
    model = getattr(models, name)(pretrained=pretrained)
    model.conv1 = torch.nn.Conv2d(6, 64, (7,7), stride=(2,2), padding=(3,3))
    model.fc = torch.nn.Linear(512, 1108)
    return model
