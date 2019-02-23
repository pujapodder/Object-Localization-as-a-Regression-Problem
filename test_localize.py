

import TestDataLoader
import Model_for_testing
import utils


with open('TestResult.txt', 'w') as file:
    file.write('')

def TestData():

    #Load Test Data
    # testLoader = test_localize.loadTestData(r'/import/helium-share/staff/ywang/comp7950/test_images.txt')
    testLoader = TestDataLoader.loadTestData(r'/import/helium-share/staff/ywang/comp7950/test_images.txt')

    #Load test model
    model = Model_for_testing.loadModel('model.pt')

    #Computing Result
    result = Model_for_testing.testModel(model, testLoader)

    #printing out the results
    it = iter(result)
    for imgs, imgSizes in testLoader:
        res = utils.box_transform_inv(next(it).data.cpu(), imgSizes.data.cpu())
        Output(res)

def Output(predictedTensors):
    #x=0
    with open('TestResult.txt', 'a+') as file:
        for ptensor in predictedTensors:
            file.write('{:.2f} {:.2f} {:.2f} {:.2f}\n'.format(ptensor[0], ptensor[1], ptensor[2], ptensor[3]))
           # print('Test ' + str(x) )
           # print('-------------')  
           # print('{:.2f} {:.2f} {:.2f} {:.2f}\n'.format(ptensor[0], ptensor[1], ptensor[2], ptensor[3]))
           # print('\n')
           # x= x+1


