
import torch
from model import buildModel


def loadModel(stateDictPath):
    model = buildModel()
    model.load_state_dict(torch.load(stateDictPath))
    model.eval()
    return model


def testModel(model, testLoader):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    model = model.to(device)

    result = []
    for images in testLoader:
        images = images[0]
        images = images.to(device)
        predictedVal = model(images)
        result.append(predictedVal)
    return result