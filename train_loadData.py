
import utils
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from DatasetReader import CUBDataset




def DataLoader(train_images, train_boxes, isSplitData=True, percentage=0.1, shuffle=True):
    dataset = CUBDataset(' ', train_images, train_boxes, True)

    if isSplitData:
        sizeof_dataset = len(dataset)
        indices = list(range(sizeof_dataset))
        num_test_images = np.int32(np.floor(sizeof_dataset * percentage))

        if shuffle:
            np.random.seed(100)
            np.random.shuffle(indices)

        train_indices, validation_indices = indices[num_test_images:], indices[:num_test_images]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(validation_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                   sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=50,
                                                        sampler=valid_sampler)

        return train_loader, validation_loader
    else:
        data_loader = DataLoader(dataset, batch_size=32)
        return data_loader
