import math
import tensorflow as tf
import numpy as np
import plotly as py
import plotly.graph_objs as go
import time


def calcMSE(Xmat, Ymat, w):
    p = np.matmul(Xmat,w) - Ymat
    n = Xmat.shape[0]
    #print(p.shape)
    currLoss = np.sum((p)**2, axis = 0, keepdims=True)/(2*n)

    return currLoss

def calcError(Xmat, Ymat, w):
    #calculate the percent error.
    #
    p = np.matmul(Xmat,w)
    n = Xmat.shape[0]
    #print(p.shape)
    #print(p[0:10])
    #prediction is 1 if greater than 0.5 and 0 otherwise.
    p = p > 0.5;

    return sum(p == Ymat)/n


def linearSGD(trainData, trainTarget, miniBatchSize, learnRate, decayCoeff):
    #Perform linear stochastic gradient descent using the loss function
    #given in part 1 of the assignment with the given miniBatchSize, learnRate and decayCoeff.


    #Step 1: Reshape the training data so instead of being NxWxH it's Nxd
    Xmat = np.reshape(trainData, (trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
    #Modify Xmat so it's Nxd+1
    biasCol = np.ones((Xmat.shape[0],1))
    Xmat = np.append(biasCol,Xmat, axis=1)

    #Step 2: initialize a W column vector of zeros that's d+1x1
    n = Xmat.shape[0]
    d = Xmat.shape[1]
    w = np.zeros((d,1))

    numIter = 20000
    loss = np.zeros(math.ceil(numIter/miniBatchSize))

    #print(Xmat.shape)
    #print(trainTarget.shape)
    #print('w shape = ', w.shape)


    for i in range(numIter):
        #choose miniBatchSize elements at random.
        batchIndices = np.arange(n)
        np.random.shuffle(batchIndices)
        batchIndices = batchIndices[:miniBatchSize]

        xBatch = Xmat[batchIndices][:]
        yBatch = trainTarget[batchIndices]

        #print(xBatch.shape)
        #print(yBatch.shape)


        #This step needs to be done on the miniBatchSize subset of trainData and trainTarget.
        #w = w - learnRate*(1/miniBatchSize*(xBatch*w - yBatch).*xBatch + decayCoeff*w)
        p1 = np.matmul(xBatch,w) - yBatch
        p2 = np.multiply(p1,xBatch)
        #print('p2 shape = ', p2.shape)
        p3 = np.sum(p2, axis=0, keepdims=True).T
        #print('p3 shape = ', p3.shape)
        lossGradient = (1/miniBatchSize*p3 + decayCoeff*w)
        #print('gradient shape = ', lossGradient.shape)
        w = w - learnRate*lossGradient
        #print('new w shape ', w.shape)


        #calculate the loss once every epoch
        if i % miniBatchSize == 0:
            currLoss = calcMSE(Xmat, trainTarget, w)
            currLoss += 0.5*decayCoeff*np.sum(w**2, axis=0, keepdims=True)

            loss[int(i/miniBatchSize)] = currLoss
            #print(i)


    return w, loss


def part1_1(trainData, trainTarget):
    #Part 1:
    #SGD with miniBatchSize = 500,
    #lambda = 0;
    #numIter = 20000
    #learningRate =[0.005, 0.001, 0.0001];
    #plot training error vs # of epochs.
    #choose best learning rate.
    #learnRate = [0.005, 0.001, 0.0001]
    learnRate = [0.01]

    for learn in learnRate:
        print('learning with rate ', learn)
        w, loss = linearSGD(trainData, trainTarget, 500, learn, 0)

        # Create a trace
        trace = go.Scatter(
            x = np.arange(len(loss)),
            y = loss,
            mode = 'markers'
        )

        data = [trace]
        plot_url = py.offline.plot(data, filename='Part 1 SGD with learning rate = {}.html'.format(learn))
        #this shows that the best learning rate is 0.005

def part1_2(trainData, trainTarget):
    #Part 2:
    #SGD with miniBatchSize = [500,1500,3500]
    #lambda = 0, numIter = 20k
    #report final training MSE for each miniBatch val.
    miniBatchSize = [500,1500,3500]

    #reshaping the training data
    Xmat = np.reshape(trainData, (trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
    biasCol = np.ones((Xmat.shape[0],1))
    Xmat = np.append(biasCol,Xmat, axis=1)

    for batchSz in miniBatchSize:
        print('With batch size ', batchSz)
        runTime = time.clock()
        w, loss = linearSGD(trainData, trainTarget, batchSz, 0.005, 0)
        runTime = time.clock() - runTime
        #note that since the decay coefficeint is 0 our loss is the MSE.

        print('Final Training MSE is ', calcMSE(Xmat,trainTarget,w), '. The training time was: ', runTime)

    #miniBatchSize: 500 --> MSE 0.01197566
    #miniBatchSize:1500 --> MSE 0.01197033
    #miniBatchSize:3500 --> MSE 0.01196847

def part1_3(trainData, trainTarget, validData, validTarget, testData, testTarget):
    #Part 3:
    #SGD with lambda  = [0,0.001,0.1, 1]
    #B = 500, learnRate = 0.005, numIter = 20k
    #report final training MSE for each miniBatch val.

    #reshaping the validation data so that I can do math to it.
    validXmat = np.reshape(validData, (validData.shape[0], validData.shape[1]*validData.shape[2]))
    biasCol = np.ones((validXmat.shape[0],1))
    validXmat = np.append(biasCol,validXmat, axis=1)

    testXmat = np.reshape(testData, (testData.shape[0], testData.shape[1]*testData.shape[2]))
    biasCol = np.ones((testXmat.shape[0],1))
    testXmat = np.append(biasCol,testXmat, axis=1)

    #reshaping the training data
    Xmat = np.reshape(trainData, (trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
    biasCol = np.ones((Xmat.shape[0],1))
    Xmat = np.append(biasCol,Xmat, axis=1)

    decayCoefficeint  = [0,0.001,0.1, 1]

    for dCoeff in decayCoefficeint:
        print('With lambda =  ', dCoeff)
        w, loss = linearSGD(trainData, trainTarget, 500, 0.005, dCoeff)
        #note that since the decay coefficeint is 0 our loss is the MSE.
        print('training accuracy for lambda = ',dCoeff,': ', calcError(Xmat, trainTarget, w))
        print('validation accuracy for lambda = ',dCoeff,': ', calcError(validXmat, validTarget, w), '\n')
        print('test accuracy for lambda = ',dCoeff,': ', calcError(testXmat, testTarget, w), '\n')


def part1_4(trainData, trainTarget, validData, validTarget, testData, testTarget):

    #reshaping the validation data so that I can do math to it.
    validXmat = np.reshape(validData, (validData.shape[0], validData.shape[1]*validData.shape[2]))
    biasCol = np.ones((validXmat.shape[0],1))
    validXmat = np.append(biasCol,validXmat, axis=1)

    #Reshaping test data.
    testXmat = np.reshape(testData, (testData.shape[0], testData.shape[1]*testData.shape[2]))
    biasCol = np.ones((testXmat.shape[0],1))
    testXmat = np.append(biasCol,testXmat, axis=1)

    #reshaping the training data
    Xmat = np.reshape(trainData, (trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
    biasCol = np.ones((Xmat.shape[0],1))
    Xmat = np.append(biasCol,Xmat, axis=1)

    runTime = time.clock()
    #calculating optimal w from the normal equation:
    #w = inv(X.T*X)*X.T*y
    w = np.linalg.solve(np.matmul(Xmat.transpose(),Xmat), np.matmul(Xmat.transpose(),trainTarget))
    runTime = time.clock() - runTime

    print('For the normal equation:\n Run time: ', runTime, '\n Training MSE:', calcMSE(Xmat, trainTarget, w))
    print('Training Accuracy: ', calcError(Xmat, trainTarget, w))
    print('Validation Accuracy: ', calcError(validXmat, validTarget, w))
    print('Test Accuracy: ', calcError(testXmat, testTarget, w))

    runTime = time.clock()
    w, loss = linearSGD(trainData, trainTarget, 500, 0.005, 0)
    runTime = time.clock() - runTime

    print('For SGD:\nRun time: ', runTime, '\nTraining MSE:', calcMSE(Xmat, trainTarget, w))
    print('Training Accuracy: ', calcError(Xmat, trainTarget, w))
    print('Validation Accuracy: ', calcError(validXmat, validTarget, w))
    print('Test Accuracy: ', calcError(testXmat, testTarget, w))


    return


if __name__ == '__main__':


    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]


    #---tensorflow setup---
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #part1_1(trainData,trainTarget)
    #part1_2(trainData,trainTarget)
    #part1_3(trainData,trainTarget, validData, validTarget, testData, testTarget)
    part1_4(trainData,trainTarget, validData, validTarget, testData, testTarget)







