import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    print('raw data',data.shape)
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
                                     data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
                                     data[rnd_idx[trBatch + validBatch+1:-1],:]
    trainTarget, validTarget, testTarget =  target[rnd_idx[1:trBatch], task], \
                                            target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
                                            target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget, task


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
    Input: NewData is a N1xd matrix (i.e. testData)
           TrainData is a N2xd matrix (i.e. trainData)
           k is number of smallest value
    Output: Index(N1xk) and responsibility matrix(N1xN2)
    '''
    l2 = D_euc(NewData,TrainData)
    #No bottom_k exists in tf, so make Euc dist negative to sort
    value,index = sess.run(tf.nn.top_k(-l2,k=k))

    # #convert the index matrix into an array of column indices and an array of row indices.
    # #print(index)
    # cols = index.ravel()
    # sz = index.shape
    # rows = np.array(sorted([i for i in range(sz[0])]*k))
    # #print(sz)
    # #print(cols)
    # #print('rows = \n', rows)
    #
    # responsibility = np.zeros(l2.shape)
    # #can now use integer array indexing to create the responsibility matrix without loops.
    # #Note: this works only in numpy, not in tensorflow.
    # responsibility[rows,cols] = 1

    return index,value


def predictLabel(trainData,trainTarget, k, newData, targetData,flag =0):
    '''
    Input: TrainData is the training data
            k is k for k-nn
            NewData is input data whose output you're predicting
            TargetData is the target for the NewData.
            All inputs are numpy arrays.
    Output: (prediction, Mean Squared error of prediction and target)
    prediction is made by taking the label of the k nearest neighbours. (whichever is most frequent).
    '''

    print('Train data: ', trainData.shape)
    print('Train target: ', trainTarget.shape)
    print('new data: ', newData.shape)

    #set all non-neighbours to 0.
    index, resp = GetResponsibility(newData, trainData, k)

    n1,d = newData.shape
    pred = np.zeros([newData.shape[0],1])

    print('resp: ', resp.shape)


    for row in range(n1):
        #the labels of the k nearest neighbours
        nbs = trainTarget[index[row,:]]
        #print('neighbours: ', nbs)
        label, idx, count = sess.run(tf.unique_with_counts(nbs))

        maxidx = np.argmax(count)
        #using majority voting over k nearest neighbours to get prediction.
        pred[row] = label[idx[maxidx]]

        #print('prediction: ', pred[row])

    #classification error is the fraction of corrrect results.
    print('pred: ', pred.shape)
    #print('target: ', targetData.shape)

    boolArray = pred.ravel() == targetData.ravel()
    print ('bool Array',boolArray,boolArray.shape)
    #print(boolArray)
    #print('boolShape = ', boolArray.shape)
    sm = np.sum(boolArray)
    #print('number of correct = ', sm)

    err = 1 - sm / (int(pred.shape[0]))
    #print('err = ', err)

    #track the information of failure case when k = 10
    #we need trainDataIndex to reshape then plot
    #trainDataIndex of 10 nn is returned by GetResponsibility()
    #newDataIndex = predIndex
    if flag:
        failedIndex = np.argmin(boolArray)
        print('failIndex', failedIndex)
        trainIndex,respForFailure = GetResponsibility(newData[failedIndex:,:],trainData,k)
        print('trainIndex',trainIndex)
        return failedIndex,trainIndex[0]

    return pred, err


if __name__ == '__main__':
    trainData, validData, testData, trainTarget, validTarget, testTarget, TASK = data_segmentation('data.npy', 'target.npy', 1)

    #---tensorflow setup---
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #for k_num in [1, 5, 10, 25, 50, 100, 200]:
    #for k_num in [10]:
        #(trainPred, trainMSE) = predictLabel(trainData, trainTarget, k_num, trainData, trainTarget)
        #print ('k = ',k_num,' training class error = {0:.5f}'.format(trainMSE))
        #
        #(testPred, testMSE) = predictLabel(trainData, trainTarget, k_num, testData, testTarget)
        #print ('k = ',k_num,' test class error = {0:.5f}'.format(testMSE))

        # (validPred, validMSE) = predictLabel(trainData, trainTarget, k_num, validData, validTarget)
        # print ('k = ',k_num,' validation class error = {0:.5f}'.format(validMSE))

    failedIdx,trainIdx = predictLabel(trainData, trainTarget, 10, validData, validTarget,flag = 1)

    plt.figure(figsize=(12,12*(5**0.5-1)/2 ))



    for i in range(10):
        plt.subplot(3,5,i+6)
        plt.imshow(trainData[trainIdx[i]].reshape(32,32)*255,cmap= 'gray')
        plt.xlabel('ID :  '+str(trainTarget[trainIdx[i]]))
        plt.subplot(3, 5, 3)
        plt.imshow(validData[failedIdx].reshape(32, 32) * 255, cmap='gray')
        plt.xlabel('ID :  ' + str(validTarget[failedIdx]))
        plt.tight_layout()
    if TASK == 0:

        plt.title('Failed Facial Recognization')
        plt.savefig('part3FFR.jpg')
    else:
        plt.title('Failed Gender Recognization')
        plt.savefig('part3FGR.jpg')




    plt.show()

