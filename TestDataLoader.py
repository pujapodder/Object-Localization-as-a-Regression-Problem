


import utils
from torch.utils.data import DataLoader
from DatasetReader import CUBDataset



def loadTestData(testImageFile):
    dataset = CUBDataset(' ', testImageFile)
    dataLoader = DataLoader(dataset, batch_size = 32)
    return dataLoader