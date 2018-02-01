
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

#---Q1 Euclidean Distance Function---
def D_euc(X,Z):
    '''
    Euclidean distance
    Input: X is a N1xd matrix
           Z is a N2xd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||X[i,:]-Z[j,:]||^2
    '''
    X_norm = tf.reshape(tf.reduce_sum(X**2, axis=1), [X.shape[0],1])
    #row vector of z norms
    Z_norm = tf.reshape(tf.reduce_sum(Z**2, axis=1), [1,Z.shape[0]])
    dist = X_norm + Z_norm-2*tf.matmul(X,tf.transpose(Z))
    return dist

#---Q2 part 1: choosing the nearest neighbours.---
def GetResponsibility(NewData,TrainData,k):
    '''
    Average nearest number
    Input: NewData is a N1xd matrix (i.e. testData)
           TrainData is a N2xd matirx (i.e. trainData)
           k is number of smallest value
    Output: Index(N1xk) and responsibility matrix(N1xN2)
    '''
    l2 = D_euc(NewData,TrainData)
    #No bottom_k exists in tf, so make Euc dist negative to sort
    value,index = sess.run(tf.nn.top_k(-l2,k=k))

    #convert the index matrix into an array of column indices and an array of row indices.
    #print(index)
    cols = index.ravel()
    sz = index.shape
    rows = np.array(sorted([i for i in range(sz[0])]*k))
    #print(sz)
    #print(cols)
    #print('rows = \n', rows)

    responsibility = np.zeros(l2.shape)
    #can now use integer array indexing to create the responsibility matrix without loops.
    #Note: this works only in numpy, not in tensorflow.
    responsibility[rows,cols] = 1/k

    return index,responsibility


def predict(trainData,trainTarget, k, newData, targetData):
    '''
    Input: TrainData is the training data
            k is k for k-nn
            NewData is input data whose output you're predicting
            TargetData is the target for the NewData.
            All inputs are numpy arrays.

    Output: (prediction, Mean Squared error of prediction and target)
    Prediction made by taking average k nearest neighbours.
    '''

    index, responsibility = GetResponsibility(newData, trainData, k)

    #prediction of the value for the new data.
    pred = np.matmul(responsibility, trainTarget)

    #mean squared error of prediction and target data
    MSE = np.sum((pred - targetData)) ** 2 / (2*int(pred.shape[0]))

    #print('new data: ', newData)
    #print('Train data: ', trainData)
    #print('target: ', targetData.shape)
    #print('resp: ', responsibility)
    #print('pred: ', pred.shape)
    #print('prediction = ', pred)
    #print('target = ', targetData)

    return pred, MSE


if __name__ == '__main__':
    #---Q2 setup, Generating the test data set. (code given)---
    np.random.seed(521)
    Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
    Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
    + 0.5 * np.random.randn(100 , 1)
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
    validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

    #---initialize newData---
    newData = np.linspace(1.0,10.0,num =1000)[:, np.newaxis]


    #---tensorflow setup---
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)


    for k_num in [1,3,5,50]:
        #change the 4th arg to do train and test
        (trainPred, trainMSE) = predict(trainData, trainTarget, k_num, trainData, trainTarget)
        print ('k = ',k_num,' training MSE = {0:.5f}'.format(trainMSE))

        (testPred, testMSE) = predict(trainData, trainTarget, k_num, testData, testTarget)
        print ('k = ',k_num,' test MSE = {0:.5f}'.format(testMSE))

        (validPred, validMSE) = predict(trainData, trainTarget, k_num, validData, validTarget)
        print ('k = ',k_num,' validation MSE = {0:.5f}'.format(validMSE))

    #plt.show()





