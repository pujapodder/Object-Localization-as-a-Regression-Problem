
import train_loadData
import training

with open('training.txt', 'w') as file:
    file.write('')


def loadData():
        DataLoader, TrainingDataLoader = train_loadData.DataLoader(r'train_images.txt',
                                                                   r'train_boxes.txt')
        training.train_model(DataLoader, TrainingDataLoader)








    
