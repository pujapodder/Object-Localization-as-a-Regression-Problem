
import train_loadData
import training

with open('training.txt', 'w') as file:
    file.write('')


def loadData():
        DataLoader, TrainingDataLoader = train_loadData.DataLoader(r'/import/helium-share/staff/ywang/comp7950/train_images.txt',
                                                                   r'/import/helium-share/staff/ywang/comp7950/train_boxes.txt')
        training.train_model(DataLoader, TrainingDataLoader)








    
