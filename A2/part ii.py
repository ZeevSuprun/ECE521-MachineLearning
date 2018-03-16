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
    X = np.concatenate((np.ones((N,1)),X),axis=1) #adding bias
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
        if i*B%N == 0:
            L2 = calcMSE(X,y,w,dCoef=lam).reshape(-1)
            losslist.append(L2)
    return losslist,w

def squared_error(labels=None,logits=None):
    loss = tf.losses.mean_squared_error(labels,logits,weights =0.5)
    err = tf.add(logits,-labels)
    return loss#1*(tf.multiply(err,err))


def miniBatch_SGD(trainData,trainTarget,validData,validTarget,lam=0.01,eta=0.001,B=500,iterNum=1000,classNum=1,lossfunc=tf.nn.sigmoid_cross_entropy_with_logits,*arg,**kwargs):
    #reshape data and targets
    X_train = trainData.reshape(len(trainData), -1)
    X_valid = validData.reshape(len(validData),-1)
    y_train = trainTarget
    N, d = X_train.shape  #len(data), len(features)
    #create label matrix based on probabilities
    #convert X and y to tensor
    y_valid = tf.convert_to_tensor(validTarget, dtype=np.float32)
    X_valid = tf.convert_to_tensor(X_valid, dtype=np.float32)
    #create Xn and yn placeholder
    yn_train = tf.placeholder(dtype=np.float32,shape=(B,classNum))
    Xn_train = tf.placeholder(dtype=np.float32,shape=(B,d))
    #create w, b, bn tf.variables
    w = tf.Variable(tf.zeros(shape=(d,classNum)))
    b = tf.Variable(tf.zeros(shape=(len(validData),classNum)))
    bn = tf.Variable(tf.zeros(shape=(B,classNum)))
    #create entire batch graph to output loss and acc
    yHat = tf.add(tf.matmul(X_valid, w), b)
    cross_entropy = tf.reduce_mean(lossfunc(labels=y_valid, logits=yHat))
    weight_penalty = tf.multiply(lam / 2., tf.reduce_sum(tf.multiply(w, w)))
    loss = tf.add(cross_entropy, weight_penalty)
    #create mini-batch graph to train
    ynHat_train = tf.add(tf.matmul(Xn_train, w), bn)
    cross_entropy = tf.reduce_mean(lossfunc(labels=yn_train, logits=ynHat_train))
    weight_penalty = tf.multiply(lam / 2., tf.reduce_sum(tf.multiply(w, w)))
    batchloss = tf.add(cross_entropy, weight_penalty)
    train = tf.train.GradientDescentOptimizer(learning_rate=eta).minimize(batchloss)
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
            batchIndices = batchIndices[:B]
            Xn = X_train[batchIndices]
            yn = y_train[batchIndices]
            #train step
            sess.run(train,feed_dict={Xn_train:Xn,yn_train:yn})
            #memorize output
            lossList.append(sess.run(loss))
            y_pred =sess.run(yHat)
            #print(y_pred.shape)
            accList.append(np.mean((abs(y_pred-validTarget))<abs(y_pred-(1-validTarget))))
    return lossList,accList,np.linspace(0,iterNum*B/float(N),len(lossList))

def miniBatch_aOp(trainData,trainTarget,validData,validTarget,lam=0.01,eta=0.001,B=500,iterNum=1000,classNum=10,lossfunc=tf.nn.softmax_cross_entropy_with_logits,*arg,**kwargs):
    #reshape data and targets
    X_train = trainData.reshape(len(trainData), -1)
    X_valid = validData.reshape(len(validData),-1)
    N, d = X_train.shape  #len(data), len(features)
    #create label matrix based on probabilities
    if classNum > 1:
        y_train = np.zeros(shape=(N,classNum))
        for i in range(N):
            y_train[i,trainTarget[i]]=1.
        y_valid = np.zeros(shape=(len(X_valid),classNum))
        for i in range(len(X_valid)):
            y_valid[i,validTarget[i]]=1.
    else:
        y_train = trainTarget
        y_valid = validTarget

    #convert X and y to tensor
    y_valid = tf.convert_to_tensor(y_valid, dtype=np.float32)
    X_valid = tf.convert_to_tensor(X_valid, dtype=np.float32)
    #create Xn and yn placeholder
    yn_train = tf.placeholder(dtype=np.float32,shape=(B,classNum))
    Xn_train = tf.placeholder(dtype=np.float32,shape=(B,d))
    #create w, b, bn tf.variables
    w = tf.Variable(tf.zeros(shape=(d,classNum)))
    b = tf.Variable(tf.zeros(shape=(len(validData),classNum)))
    bn = tf.Variable(tf.zeros(shape=(B,classNum)))
    #create entire batch graph to output loss and acc
    yHat = tf.add(tf.matmul(X_valid, w), b)
    cross_entropy = tf.reduce_mean(lossfunc(labels=y_valid, logits=yHat))
    weight_penalty = tf.multiply(lam / 2., tf.reduce_sum(tf.multiply(w, w)))
    loss = tf.add(cross_entropy, weight_penalty)

    y_pred = tf.argmax(yHat,axis=1)

    #create mini-batch graph to train
    ynHat_train = tf.add(tf.matmul(Xn_train, w), bn)
    cross_entropy = tf.reduce_mean(lossfunc(labels=yn_train, logits=ynHat_train))
    weight_penalty = tf.multiply(lam / 2., tf.reduce_sum(tf.multiply(w, w)))
    batchloss = tf.add(cross_entropy, weight_penalty)
    train = tf.train.AdamOptimizer(learning_rate=eta).minimize(batchloss)
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
            batchIndices = batchIndices[:B]
            Xn = X_train[batchIndices]
            yn = y_train[batchIndices]
            #train step
            sess.run(train,feed_dict={Xn_train:Xn,yn_train:yn})
            #print(np.mean(abs(sess.run(yHat)-validTarget)<abs(sess.run(yHat)-(1-validTarget))))
            #memorize output
            lossList.append(sess.run(loss))
            if classNum>1:
                accList.append(np.mean(sess.run(y_pred) == validTarget))
            else:
                accList.append(np.mean(abs(sess.run(yHat)-validTarget)<abs(sess.run(yHat)-(1-validTarget))))
        print(len(lossList),len(accList))

    return lossList,accList,np.linspace(0,iterNum*B/float(N),len(lossList))

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
            loss = linearSGD(trainData, trainTarget, eta=learnRate)[0]
            plt.plot(loss,label ='learning Rate = '+str(learnRate))
            plt.xlabel('number of epochs')
            plt.ylabel('train loss')
            plt.xlim(0,len(loss))
            plt.legend(fontsize='small')
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
         validData = validData.reshape(len(validData), -1)
         testData = testData.reshape(len(testData), -1)

         X_valid = np.concatenate((np.ones((len(validData), 1)), validData), axis=1)  # adding bias
         X_test = np.concatenate((np.ones((len(testData), 1)), testData), axis=1)


         for decayCoef in [0.,0.001,0.1,1]:

             w = linearSGD(trainData,trainTarget,lam=decayCoef)[1]

             yHat_valid = np.matmul(X_valid,w)
             acc_valid = np.mean(abs(yHat_valid-validTarget)<abs(yHat_valid-(1-validTarget)))
             yHat_test = np.matmul(X_test, w)
             acc_test = np.mean(abs(yHat_test - testTarget) < abs(yHat_test - (1 - testTarget)))


             print("validation accuracy= ", acc_valid,"test accuracy= ",acc_test,'when lambda = ',decayCoef)
#1.4
    if Q14:
        print('---Q14---')
        trainData = trainData.reshape(len(trainData), -1)
        trainData = np.concatenate((np.ones((len(trainData), 1)), trainData), axis=1)

        validData = validData.reshape(len(validData), -1)
        validData = np.concatenate((np.ones((len(validData), 1)), validData), axis=1)

        testData = testData.reshape(len(testData), -1)
        testData = np.concatenate((np.ones((len(testData), 1)), testData), axis=1)

        # closed form LMS solution
        start = time.time()
        w_opt = np.linalg.solve(np.matmul(trainData.transpose(), trainData), np.matmul(trainData.transpose(), trainTarget))
        end = time.time()
        elapse = end - start

        train_loss = calcMSE(trainData, trainTarget, w_opt)
        valid_loss = calcMSE(validData,validTarget,w_opt)
        test_loss = calcMSE(testData,testTarget,w_opt)

        def Acc(X,y):
            yHat = np.matmul(X,w_opt)
            return np.mean(abs(yHat-y)<abs(yHat-(1-y)))

        train_acc = Acc(trainData,trainTarget)
        valid_acc = Acc(validData,validTarget)
        test_acc = Acc(testData,testTarget)

        print('Computation time ',elapse, 's')
        print('train loss',train_loss,'train acc',train_acc )
        print('valid loss',valid_loss,'valid acc',valid_acc)
        print('test loss',test_loss,'test acc',test_acc)

#2.1.1
    if Q211:
        print('---Q211---')
        for learnRate in [0.005]:
            valid_loss, valid_acc, valid_epoch = miniBatch_SGD(trainData, trainTarget, validData, validTarget,
                                                               B=500, lam=0.01, eta=learnRate, classNum=1, iterNum=5000,
                                                               lossfunc=tf.nn.sigmoid_cross_entropy_with_logits)
            train_loss, train_acc, train_epoch = miniBatch_SGD(trainData, trainTarget, trainData, trainTarget,
                                                               B=500, lam=0.01, eta=learnRate, classNum=1, iterNum=5000,
                                                               lossfunc=tf.nn.sigmoid_cross_entropy_with_logits)
            print('learnRate=', learnRate, 'train loss', np.mean(train_loss[-20:]), 'train acc',
                  np.mean(train_acc[-20:]))
            plt.figure()
            plt.subplot(211)
            plt.title('tuned learning rate=' + str(learnRate))
            plt.plot(train_epoch, train_loss, label='training cross-entropy loss')
            plt.plot(valid_epoch, valid_loss, label='validation cross-ntropy loss')
            plt.ylabel('regularized cross entropy loss')
            plt.xlim(0,valid_epoch[-1])
            plt.legend(fontsize='small')
            plt.subplot(212)
            print(train_epoch.shape)
            plt.plot(train_epoch, train_acc, label='training classification accuracy')
            plt.plot(valid_epoch, valid_acc, label='validation classification accuracy')
            plt.xlim(0, valid_epoch[-1])
            plt.xlabel('number of epochs')
            plt.ylabel('classification accuracy')
            plt.legend(loc='lower right', fontsize='small')

        plt.savefig('Q211.jpg')
        plt.show()

    if Q212:
        print('---Q212---')
        train_loss_SGD, train_acc_SGD, train_epoch_SGD = miniBatch_SGD(trainData, trainTarget, trainData, trainTarget,
                                                            B=500, lam=0.01, eta=0.001, classNum=1, iterNum=5000,
                                                            lossfunc=tf.nn.sigmoid_cross_entropy_with_logits)

        train_loss_aOp, train_acc_aOp, train_epoch_aOp = miniBatch_aOp(trainData, trainTarget, trainData, trainTarget,
                                                            B=500, lam=0.01, eta=0.001, classNum=1, iterNum=5000,
                                                            lossfunc=tf.nn.sigmoid_cross_entropy_with_logits)
        plt.figure()
        plt.plot(train_epoch_SGD, train_loss_SGD, label='SGD training cross-entropy loss')
        plt.plot(train_epoch_aOp, train_loss_aOp, label='AdamOptimizer training cross-entropy loss')
        plt.ylabel('regularized cross entropy loss')
        plt.xlabel('number of epochs')
        plt.xlim(0,train_epoch_aOp[-1])

        plt.legend(fontsize='small')
        plt.savefig('Q212.jpg')
        plt.show()

    if Q213:
        print('---Q213---')

        train_loss, train_acc, train_epoch = miniBatch_aOp(trainData, trainTarget, trainData, trainTarget, lam=0.0,
                                                           classNum=1,eta=0.001,iterNum=3000,
                                                           lossfunc=tf.nn.sigmoid_cross_entropy_with_logits)
        # test_loss, test_acc, test_epoch = miniBatch_aOp(trainData, trainTarget, testData, testTarget, lam=0.0,
        #                                                    classNum=1,eta=0.001,iterNum=5000,
        #                                                    lossfunc=tf.nn.sigmoid_cross_entropy_with_logits)
        # valid_loss, valid_acc, valid_epoch = miniBatch_aOp(trainData, trainTarget, validData, validTarget, lam=0.0,
        #                                                    classNum=1,eta=0.001,iterNum=5000,
        #                                                    lossfunc=tf.nn.sigmoid_cross_entropy_with_logits)
        # print('best logistic aOp acc comparision with Q14')
        # print('train acc',train_acc[-1])
        # print('valid acc',valid_acc[-1])
        # print('test acc',test_acc[-1])

        #convergence
        LMS_loss,LMS_acc,LMS_epoch = miniBatch_aOp(trainData, trainTarget,trainData,trainTarget, lam=0.5,classNum=1,
                                       lossfunc=squared_error,B=500,eta=0.001,iterNum=3000)

        #explanation
        yHat = np.linspace(0,1,1000)
        ydummy = np.zeros(shape=yHat.shape)
        with tf.Session() as sess:
            cross_entropy_loss = -np.log(1-yHat)
            squared_error_loss = sess.run((squared_error(labels=ydummy,logits=yHat)))

        plt.figure(1)
        plt.subplot(211)
        plt.plot(train_epoch,train_loss, label='cross entropy loss')
        plt.plot(LMS_epoch,LMS_loss, label='squared-error loss')
        plt.ylabel('loss')
        plt.xlim(0,LMS_epoch[-1])
        plt.legend(fontsize='small')
        plt.subplot(212)
        plt.plot(train_epoch, train_acc, label='logistic regression training accuarcy')
        plt.plot(LMS_epoch, LMS_acc, label='linear regression training accuarcy')
        plt.xlabel('number of epoches')
        plt.ylabel('training accuarcy')
        plt.xlim(0, LMS_epoch[-1])
        plt.ylim(0,1)
        plt.legend(loc='lower right',fontsize='small')
        plt.savefig('Q213.jpg')
        # plt.figure(2)
        # plt.subplot(221)
        # plt.plot(yHat,squared_error_loss,label='squared-error loss')
        # plt.plot(yHat,cross_entropy_loss,label='cross-entropy loss')
        # plt.xlabel('yHat')
        # plt.ylabel('squared-error/cross-entropy loss')
        # plt.legend(loc='upper left',fontsize='small')
        # plt.subplot(222)
        # plt.plot(squared_error_loss,cross_entropy_loss)
        # plt.xlabel('squared-error loss')
        # plt.ylabel('cross-entropy loss')
        # plt.savefig('Q213kit3.jpg')

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
        for learnRate in [0.0001,0.001,0.005]:
            valid_loss,valid_acc,valid_epoch = miniBatch_aOp(trainData,trainTarget,validData,validTarget,
                                                         B=500,lam=0.01,eta=learnRate,classNum=10,iterNum=5000,
                                                         lossfunc=tf.nn.softmax_cross_entropy_with_logits)
            train_loss, train_acc, train_epoch = miniBatch_aOp(trainData, trainTarget, trainData, trainTarget,
                                                           B=500,lam=0.01,eta=learnRate,classNum=10,iterNum=5000,
                                                           lossfunc=tf.nn.softmax_cross_entropy_with_logits)
            print('learnRate=',learnRate,'train loss',np.mean(train_loss[-20:]),'train acc',np.mean(train_acc[-20:]))
            plt.figure()
            plt.subplot(211)
            plt.title('learning rate='+str(learnRate))
            plt.plot(train_epoch,train_loss,label='training cross-entropy loss')
            plt.plot(valid_epoch,valid_loss,label='validation cross-entropy loss')
            plt.xlim(0,valid_epoch[-1])
            plt.ylabel('regularized cross entropy loss')
            plt.legend(fontsize='small')
            plt.subplot(212)
            plt.plot(train_epoch,train_acc,label='training classification accuracy')
            plt.plot(valid_epoch,valid_acc,label='validation classification accuracy')
            plt.xlabel('number of epochs')
            plt.ylabel('classification accuracy')
            plt.ylim(0,1)
            plt.xlim(0,valid_epoch[-1])
            plt.legend(loc='lower right',fontsize='small')

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
        #tuning
        for dCoef in [0.01]:#[0,0.001,0.01,0.1,1]:
            for learnRate in [0.0001]:#[0.005,0.001,0.0001]:
                valid_loss, valid_acc, valid_epoch = miniBatch_aOp(trainData, trainTarget, validData, validTarget,
                                                                   B=300,classNum=6,lam=dCoef,eta=learnRate,iterNum=20000,
                                                                   lossfunc=tf.nn.softmax_cross_entropy_with_logits)
                train_loss, train_acc, train_epoch = miniBatch_aOp(trainData, trainTarget, trainData, trainTarget,iterNum=20000,
                                                                   B=300,classNum=6,lam=dCoef,eta=learnRate,
                                                                   lossfunc=tf.nn.softmax_cross_entropy_with_logits)
                print('learnRate=', learnRate, 'decay',dCoef,'valid loss', np.mean(valid_loss[-20:]), 'valid acc',
                      np.mean(valid_acc[-20:]))
                plt.figure(1)
                plt.subplot(211)
                plt.title('weight decay coefficient=' + str(dCoef) + ';learning rate=' + str(learnRate))
                plt.plot(train_epoch, train_loss, label='training cross-entropy loss')
                plt.plot(valid_epoch, valid_loss, label='validation cross-entropy loss')
                plt.xlabel('number of epochs')
                plt.ylabel('regularized cross-entropy loss')
                plt.xlim(0,valid_epoch[-1])
                plt.legend(fontsize='small')
                plt.subplot(212)
                plt.plot(train_epoch, train_acc, label='training classification accuracy')
                plt.plot(valid_epoch, valid_acc, label='validation classification accuracy')
                plt.xlim(0,valid_epoch[-1])
                plt.ylim(0,1)
                plt.xlabel('number of epochs')
                plt.ylabel('classification accuracy')
                plt.legend(loc='lower right', fontsize='small')

        plt.savefig('Q222.jpg')
        plt.show()







