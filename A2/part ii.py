###ECE 521 A2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def calcMSE(Xmat,Ymat,weight,dCoef=0):
    p = np.matmul(Xmat,weight) - Ymat
    n = Xmat.shape[0]
    return np.sum((p)**2,axis=0)/(2*n)+dCoef/2.*np.sum(weight**2,axis=0)

def linearSGD(trainData,trainTarget,eta=0.005,B=500,lam=0,iterNum=20000):
    X = trainData.reshape(trainData.shape[0],-1) #make mini-batch
    N,d = X.shape  # num of data, num of features
    X = np.concatenate((np.ones((n,1)),X),axis=1) #adding bias
    y = trainTarget
    w = np.zeros(shape=(d+1,1))
    losslist = []   #create loss list
    for i in range(iterNum):
        batchIndices = np.arange(N)
        np.random.shuffle(batchIndices)
        batchIndices = batchIndices[:B]
        Xn = X[batchIndices]
        yn = y[batchIndices]
        grad = np.matmul(Xn.transpose(),(np.matmul(Xn,w)-yn))/B + lam*w
        w -= eta*grad
        if i*B%n == 0:
            L2 = calcMSE(X,y,w,dCoef=lam).reshape(-1)
            losslist.append(L2)
    return losslist,w

def logisticSGD(trainData,trainTarget,eta=0.001,B=500,lam=0.01,iterNum=1000):
    X = trainData.reshape(len(trainData),-1) #make mini-batch
    N,d = X.shape  # num of data, num of features
    X = np.concatenate((np.ones((n,1)),X),axis=1) #adding bias
    y = trainTarget
    w = np.zeros(shape=(d+1,1))
    losslist = []   #create loss2 list
    for i in range(iterNum):
        batchIndices = np.arange(N)
        np.random.shuffle(batchIndices)
        batchIndices = batchIndices[:B]
        Xn = X[batchIndices]   #reshape Xi as a column vector
        yn = y[batchIndices]
        Xnw = np.matmul(Xn,w)
        grad = np.matmul(Xn.transpose(),(1-yn-1/(1+np.exp(Xnw))))/B + lam*w
        w -= eta*grad
        #append when integer epoch
        if i*B%n == 0:
            y_tensor = tf.convert_to_tensor(y, dtype=np.float32)
            Xw_tensor = tf.convert_to_tensor(np.matmul(X,w), dtype=np.float32)
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_tensor,logits=Xw_tensor))
            losslist.append(sess.run(loss)/n+lam/2.*np.sum(w**2,axis=0))
    return losslist,w

def squared_error(labels=None,logits=None):
    err = tf.add(logits,-labels)
    return 0.5*(tf.multiply(err,err))

def miniBatch_aOp(trainData,trainTarget,lam=0.01,B=500,iterNum=100,classNum=10,lossfunc=tf.nn.softmax_cross_entropy_with_logits_v2,*arg,**kwargs):
    #reshape data and targets
    X = trainData.reshape(len(trainData), -1)
    y = trainTarget
    N, d = X.shape  #len(data), len(features)
    #create true label matrix based on probabilities
    if classNum > 1:
        y = np.zeros(shape=(N,classNum))
        for i in range(N):
            y[i,trainTarget[i]]=1.
    #convert X and y to tensor
    y_tensor = tf.convert_to_tensor(y, dtype=np.float32)
    X_tensor = tf.convert_to_tensor(X, dtype=np.float32)
    #create Xn and yn placeholder
    yn_tensor = tf.placeholder(dtype=np.float32,shape=(B,y.shape[-1]))
    Xn_tensor = tf.placeholder(dtype=np.float32,shape=(B,X.shape[-1]))
    #create w, b, bn tf.variables
    w_tensor = tf.Variable(tf.zeros(shape=(d,classNum)))
    b_tensor = tf.Variable(tf.zeros(shape=(N,classNum)))
    bn_tensor = tf.Variable(tf.zeros(shape=(B,classNum)))
    #create entire batch graph to output loss and acc
    yHat_tensor = tf.add(tf.matmul(X_tensor, w_tensor), b_tensor)
    cross_entropy = tf.reduce_mean(lossfunc(labels=y_tensor, logits=yHat_tensor))
    weight_penalty = tf.multiply(lam / 2., tf.reduce_sum(tf.multiply(w_tensor, w_tensor)))
    loss = tf.add(cross_entropy, weight_penalty)
    yHat = tf.argmax(yHat_tensor,axis=1)
    #create mini-batch graph to train
    ynHat_tensor = tf.add(tf.matmul(Xn_tensor, w_tensor), bn_tensor)
    cross_entropy = tf.reduce_mean(lossfunc(labels=yn_tensor, logits=ynHat_tensor))
    weight_penalty = tf.multiply(lam / 2., tf.reduce_sum(tf.multiply(w_tensor, w_tensor)))
    batchloss = tf.add(cross_entropy, weight_penalty)
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(batchloss)
    #initialize and then evaluate graph step by step
    init = tf.global_variables_initializer()
    lossList=[]
    accList=[]
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterNum):
            #shuffle batch
            batchIndices = np.arange(N)
            np.random.shuffle(batchIndices)
            batchIndices = batchIndices[:500]
            Xn = X[batchIndices]
            yn = y[batchIndices]
            #train step
            sess.run(train,feed_dict={Xn_tensor:Xn,yn_tensor:yn})
            #memorize output
            lossList.append(sess.run(loss))
            accList.append(np.mean(sess.run(yHat) == trainTarget))
    return lossList,accList

def notMNIST_2c(Q11=0,Q12=0,Q13=0,Q14=0,Q211=0,Q212=0,Q213=0):
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
    print('---Q1~Q213 raw data---')
    print('trainData shape',trainData.shape)
    print('validData shape',validData.shape)
    print('testData shape',testData.shape)
#1.1
    if Q11:
        print('---Q11---')
        plt.figure()
        for learnRate in [0.005,0.001,0.0001]:
            plt.plot(linearSGD(trainData,trainTarget,eta=learnRate)[0],\
                     label ='learning Rate = '+str(learnRate))
            plt.xlabel('epoch')
            plt.ylabel('train loss')
            plt.legend()
        plt.savefig('Q11.jpg')
        plt.show()
#1.2
    if Q12:
        print('---Q12---')
        for batchSize in [500,1500,3500]:
            start = time.time()
            MSE = linearSGD(trainData,trainTarget,eta=0.005,B=batchSize)[0][-1]# -1 output the last MSE
            end = time.time()
            elapse = end - start
            print('with batch size ', batchSize, 'final trainMSE ',MSE,' train time ',elapse,'s')
#1.3
    if Q13:
         print('---Q13---')
         X = validData.reshape(len(validData), -1)
         n,d = X.shape  # num of data, num of features
         X = np.concatenate((np.ones((n, 1)), X), axis=1)  # adding bias
         y = validTarget

         for decayCoef in [0.,0.001,0.1,1]:
             w = linearSGD(trainData,trainTarget,lam=decayCoef)[1]
             L2 = calcMSE(X,y,w,dCoef=decayCoef)
             print(L2)
        #To do test MSE
#1.4
    if Q14:
        print('---Q14---')
        start  = time.time()
        X = trainData.reshape(len(trainData), -1)
        n,d = X.shape  # num of data, num of features
        X = np.concatenate((np.ones((n,1)),X),axis=1)  # adding bias
        y = trainTarget
        #closed form LMS solution
        w_opt = np.linalg.solve(np.matmul(X.transpose(),X), np.matmul(X.transpose(),y))
        L2 = calcMSE(X,y,w_opt)
        end = time.time()
        elapse = end - start
        print('Closed form MSE ',L2,' time ',elapse, 's')
#2.1.1
    if Q211:
        print('---Q211---')
        plt.figure()
        for learnRate in [0.005,0.001,0.0001]: #tuning eta??
            plt.plot(logisticSGD(trainData, trainTarget,eta=learnRate)[0],label='learning Rate = ' + str(learnRate))
            plt.xlabel('epoch')
            plt.ylabel('train loss')
            plt.legend()
        plt.savefig('Q211.jpg')
        plt.show()
    if Q212:
        print('---Q212---')
        sigmoidLoss,acc = miniBatch_aOp(trainData,trainTarget,classNum=1,lossfunc=tf.nn.sigmoid_cross_entropy_with_logits)
        SGD_cross_entropy_loss = logisticSGD(trainData, trainTarget,eta=0.001)[0]
        plt.figure()
        plt.plot(sigmoidLoss,label='adam optimizer cross entropy loss')
        plt.plot(SGD_cross_entropy_loss,label='SGD cross entropy loss')
        plt.legend()
        plt.savefig('Q212.jpg')
        plt.show()
    if Q213:
        print('---Q213---')
        #convergence
        norm2loss,accuary = miniBatch_aOp(trainData, trainTarget, lam=0,classNum=1,
                                       lossfunc=squared_error)
        sigmoidLoss, acc = miniBatch_aOp(trainData, trainTarget,lam=0.01, classNum=1,
                                       lossfunc=tf.nn.sigmoid_cross_entropy_with_logits)
        #explanation
        yHat = np.linspace(0,1,1000)
        ydummy = np.zeros(shape=yHat.shape)
        with tf.Session() as sess:
            cross_entropy_loss = sess.run((tf.nn.sigmoid_cross_entropy_with_logits(labels=ydummy, logits=yHat)))
            squared_error_loss = sess.run((squared_error(labels=ydummy,logits=yHat)))

        plt.figure()
        plt.subplot(211)
        plt.plot(sigmoidLoss, label='sigmoid cross entropy loss')
        plt.plot(norm2loss, label='squared-error loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.subplot(212)
        plt.plot(yHat,cross_entropy_loss,label='sigmoid cross entropy loss')
        plt.plot(yHat,squared_error_loss,label='squared_error loss')
        plt.xlabel('predicted target(true target = 0)')
        plt.legend()
        plt.savefig('Q213.jpg')

        plt.show()


def notMNIST_10c(Q221):
    if Q221:
        with np.load("notMNIST.npz") as data:
            Data, Target = data["images"], data["labels"]
            np.random.seed(521)
            randIndx = np.arange(len(Data))
            np.random.shuffle(randIndx)
            Data = Data[randIndx] / 255.
            Target = Target[randIndx]
            trainData, trainTarget = Data[:15000], Target[:15000]
            validData, validTarget = Data[15000:16000], Target[15000:16000]
            testData, testTarget = Data[16000:], Target[16000:]
        print('---Q221 raw data---')
        print('trainData shape', trainData.shape,'trainTarget shape', trainTarget.shape)
        print('validData shape', validData.shape, 'validTarget shape', validTarget.shape)
        print('testData shape', testData.shape, 'testTarget shape', testTarget.shape)

        softmaxLoss,softmaxAcc = miniBatch_aOp(trainData,trainTarget,classNum=10,lossfunc=tf.nn.softmax_cross_entropy_with_logits_v2)
        plt.figure()
        plt.plot(softmaxLoss,label='train cross entropy loss')
        plt.plot(softmaxAcc,label='validation accuarcy')
        plt.xlabel('epoch')
        #plt.ylim(ymin=0)
        plt.legend()
        plt.savefig('Q221.jpg')
        plt.show()

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
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def faceScrub_6c(Q222):
    if Q222:
        trainData, validData, testData, trainTarget, validTarget, testTarget \
            = data_segmentation('data.npy','target.npy', 0)
        print('---Q222 raw data---')
        print('trainData shape', trainData.shape, 'trainTarget shape', trainTarget.shape)
        print('validData shape', validData.shape, 'validTarget shape', validTarget.shape)
        print('testData shape', testData.shape, 'testTarget shape', testTarget.shape)

        softmaxLoss, softmaxAcc = miniBatch_aOp(trainData, trainTarget, classNum=10,\
                                              lossfunc=tf.nn.softmax_cross_entropy_with_logits_v2)
        plt.figure()
        plt.plot(softmaxLoss, label='train cross entropy loss')
        plt.plot(softmaxAcc, label='validation accuarcy')
        plt.xlabel('epoch')
        plt.ylim(ymin=0)
        plt.legend()
        plt.savefig('Q222.jpg')
        plt.show()

if __name__ == '__main__':
    #---tensorflow setup---
    sess = tf.Session()
    notMNIST_2c(Q11=0,Q12=0,Q13=0,Q14=0,Q211=0,Q212=0,Q213=1)
    notMNIST_10c(0)
    faceScrub_6c(0)





