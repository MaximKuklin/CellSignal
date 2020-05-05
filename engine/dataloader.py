import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CellDataset(Dataset):
    def __init__(self, root, mode, transforms=None, show=False):

        path = os.path.join(root, 'annotations', mode+'.csv')
        self.mode = mode
        if self.mode == 'test':
            self.images_path = os.path.join(root, self.mode)
        else:
            self.images_path = os.path.join(root, 'train')
        self.transforms = transforms
        self.data = pd.read_csv(path).values
        classes = sorted([int(elem[6:]) for elem in np.unique(self.data[..., -1])])
        classes_interpr = np.arange(1108)
        self.classes = {'sirna_'+str(classes[i]):classes_interpr[i] for i in range(1108)}
        self.show = show

    def __len__(self):
        return len(self.data)

    def get_image(self, row):
        path = os.path.join(row[1], 'Plate'+str(row[2]))
        full_image = []
        for level in range(1, 7):
            image_name = '_'.join([row[3], 's'+str(row[4]), 'w'+str(level)])+'.png'
            image_name = os.path.join(self.images_path, path, image_name)
            image = np.asarray(Image.open(image_name))
            full_image.append(image)
        full_image = np.stack(full_image)
        return full_image

    def __getitem__(self, idx):
        line = self.data[idx]
        image = self.get_image(line)

        if self.transforms is not None:
            image = image.transpose(1, 2, 0)
            image = self.transforms(image=image)['image']

        if self.show:
            plt.imshow(image[3:].transpose(1,2,0))
            plt.show()

        cls = int(self.classes[line[-1]])
        return image, cls


