import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CUBDataset(Dataset):

    def __init__(self, split, id_file_path, bound_file_path='', is_training=False, transform=None):
        self.images = []
        self.isTraining = is_training

        with open(id_file_path) as idFile:
            idFiles = idFile.readlines()

        if is_training:
            with open(bound_file_path) as boxFile:
                boundingBoxes = boxFile.readlines();

        for ind in range(len(idFiles)):

            imageFileName = idFiles[ind].strip()
            currentDir = os.path.dirname(__file__)
            imageFileName = os.path.join(currentDir, r'images', imageFileName)
            if is_training:
                self.images.append((imageFileName, np.array(boundingBoxes[ind].strip().split(split), dtype=np.float32)))
            else:
                self.images.append(imageFileName)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),

            ])

    def __getitem__(self, index):
        if self.isTraining:

            imagePath, boundingBox = self.images[index]

            image = Image.open(imagePath).convert('RGB')
            imageSize = np.array(image.size, dtype=np.float32)
            image = self.transform(image)

            return image, boundingBox, imageSize
        else:
            imagePath = self.images[index]
            image = Image.open(imagePath).convert('RGB')
            imageSize = np.array(image.size, dtype=np.float32)
            image = self.transform(image)

            return image, imageSize

    def __len__(self):
        return len(self.images)
