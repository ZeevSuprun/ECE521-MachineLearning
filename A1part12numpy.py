import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#---Q1 Euclidean Distance Function---
def D_euc(X,Z):
    '''
    Input: X is a N1xd matrix
           Z is a N2xd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||X[i,:]-Z[j,:]||^2
    '''
    X_norm = (X**2).sum(axis=1).reshape(X.shape[0],1)
    Z_norm = (Z**2).sum(axis=1).reshape(1,Z.shape[0])
    dist = X_norm+Z_norm-2*X.dot(Z.transpose())
    return dist

#---Q2 Making Prediction for Regression---
def cnn(NewData,TrainData,k):
    '''Choosing nearest number
    Input: NewData is a N1xd matrix (i.e. testData)
           TrainData is a N2xd matirx (i.e. trainData)
           k is number of smallest value
    Output: Index(N1xk) and responsibility matrix(N1xN2)
    '''
    l2 = D_euc(NewData,TrainData)
    print ('Euc dist shape is :',l2.shape)
    #No bot_k exists in tf, so make Euc dist negative to sort
    value,index = sess.run(tf.nn.top_k(-l2,k=k))
    #res = tf.zeros(tf.shape(l2), tf.int32)
    #res = tf.SparseTensorValue(indices = b, values = a, dense_shape = tf.shape(l2))
    res = np.zeros(l2.shape)
    rown,coln = index.shape
    print (rown,coln)
    for j in range(coln):
        for i in range(rown):

            aidx = index[i,j]
            res[i,aidx] = 1/k
    return index,res

#---Q2 setup, codes given---
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

#---run Q2 ---
def Q2_run(NewData,TrainData,k,XTarget,TrainTarget,flag = 0):
    idx, Responsibility = cnn(NewData, TrainData, k)
    Pred = Responsibility @ TrainTarget
    # MSE = sum((pred - Xtarget)) ** 2 / (2 * len(pred))
    # print ('k = ',k,' MSE = ',np.asscalar(MSE))
    if flag:

        plt.figure()
        plt.plot(trainData, trainTarget, 'bo',label = 'train')
        plt.plot(validData, validTarget, 'ro',label = 'valid')
        plt.plot(testData,testTarget,'yo',label = 'test')
        plt.plot(NewData, Pred, 'g.',label = 'prediction')
        plt.title('k = '+str(k))
        plt.legend(loc = 2)

# print(trainData.shape)
# print(testData.shape)
# print ('idx',idx.shape,idx)
# print ('resp',resp.shape,resp)
# print(pred)
# print(testTarget)

#---add to main---
for k_num in [1,3,5,50]:
    #change the 4th arg to do train and test 
    Q2_run(newData,trainData,k_num,validTarget,trainTarget,flag = 1)

plt.show()


