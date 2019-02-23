import utils
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
import torch
from model import buildModel
import Model_for_testing

#maximum epoch value
MAX_EPOCH = 80


def train_model(data_loader, training_data_loader):
     #building a model
    model = buildModel()

    #choosing between cpu and cude
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    model = model.to(device)

    criterion = nn.SmoothL1Loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_model_state = model.state_dict()
    best_epoch = -1
    best_acc = 0.0
    min_Loss = np.inf
    for epoch in range(MAX_EPOCH):
        running_loss = 0.0
        for images, boxes, im_sizes in data_loader:
            images = images.to(device)

            ##transforming box using utils function
            boxes = utils.box_transform(boxes, im_sizes)
            boxes = boxes.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            #loss calculation
            loss = criterion(outputs, boxes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        result = Model_for_testing.testModel(model, training_data_loader)
        currentAccuracy = 0
        it = iter(training_data_loader)
        for ind in range(len(result)):
            _, testBoxes, testBoxSizes = next(it)
            testBoxes = utils.box_transform(testBoxes, testBoxSizes)
            currentAccuracy += utils.compute_acc(result[ind].data.cpu(),
                                                 testBoxes.data.cpu(), testBoxSizes)
        #Calcualting Current Accuracy
        currentAccuracy = currentAccuracy / len(result)


        #Calculating Best Accuracy, Best epoch, Minimum Loss
        if best_acc < currentAccuracy:
            best_acc = currentAccuracy
            min_Loss = running_loss
            best_epoch = epoch
        #Findinf Best model State
            best_model_state = model.state_dict()
        with open('training.txt', 'a+') as file:
            file.write('current epoch: {}, current loss: {} current accuracy {:.2f} '.format(epoch, running_loss, currentAccuracy))
            file.write('best epoch: {}, best loss: {} best accuracy {:.2f}\n'.format(best_epoch, min_Loss, best_acc))

    torch.save(best_model_state, 'model.pt')
