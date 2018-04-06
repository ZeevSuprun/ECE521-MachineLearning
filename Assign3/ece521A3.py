###ECE 521 A3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def set_layer(X_tensor,d_hidden=1000):
    """
    layer-wise building block
    :param X_tensor: input tf.tensor, N by d_(L-1)
    :param d_hidden: number of the hidden units, d_(L)
    :return: weighted sum of input, signals: S = N by d_(L)
    """
    d_input = X_tensor.get_shape().as_list()[-1]
    std = (3./(d_input+d_hidden))**0.5
    W = tf.Variable(tf.random_normal([d_input,d_hidden],stddev=std),name='weight')
    b = tf.Variable(tf.zeros(d_hidden),name='bias')
    return tf.add(tf.matmul(X_tensor, W), b),W

def do_dropout(layer,dropout=1):
    if dropout:
        return tf.nn.dropout(layer, 0.5)
    else:
        return layer

with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]
    print('---Data reshaped---')
    trainData = trainData.reshape(len(trainData), -1)
    validData = validData.reshape(len(validData), -1)
    testData = testData.reshape(len(testData), -1)
    data_num, input_num = trainData.shape
    trainLabel = np.zeros(shape=(len(trainData), 10))
    for i in range(len(trainData)):
        trainLabel[i, trainTarget[i]] = 1.
    validLabel = np.zeros(shape=(len(validData), 10))
    for i in range(len(validData)):
        validLabel[i, validTarget[i]] = 1.
    testLabel = np.zeros(shape=(len(testData), 10))
    for i in range(len(testData)):
        testLabel[i, testTarget[i]] = 1.
    print('trainData shape', trainData.shape, 'trainTarget shape', trainTarget.shape)
    print('validData shape', validData.shape, 'validTarget shape', validTarget.shape)
    print('testData shape', testData.shape, 'testTarget shape', testTarget.shape)

def neural_net(xxData,xxLabel,epoch_num=50,batch_size=500,learn_rate = 0.01,weight_decay=0,
               class_num=10,d_hidden=1000,hidden_layer_num=1,dropout=0,visualization=0):
    #define placeholders
    X = tf.placeholder(tf.float32,[None,trainData.shape[-1]])
    Y = tf.placeholder(tf.float32,[None,class_num])
    #setup graph
    layer_1,W1 = set_layer(X,d_hidden=d_hidden)
    layer_1 = tf.nn.relu(layer_1)

    if hidden_layer_num>=2:     #add 2nd hidden layer if needed
        layer_2,W2 = set_layer(layer_1,d_hidden=d_hidden)
        layer_inter = tf.nn.relu(layer_2)
    else:
        layer_inter = layer_1

    layer_out,W_out = set_layer(layer_inter,d_hidden=class_num)
    layer_out=do_dropout(layer_out,dropout=dropout)
    pred_class = tf.argmax(layer_out,axis=1)
    true_class = tf.argmax(Y,axis=1)
    weight_ele_num = tf.size(W1,out_type=tf.float32)+tf.size(W_out,out_type=tf.float32)
    weight_penalty = tf.divide(weight_decay*(tf.reduce_sum(tf.multiply(W1,W1))+tf.reduce_sum(tf.multiply(W_out,W_out))),weight_ele_num )
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_out,labels=Y))
    loss = cross_entropy+weight_penalty
    train = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred_class,true_class),tf.float32))
    #eval graph
    with tf.Session() as sess:
        p25,p50,p75 = 0,0,0
        loss_list = []
        err_list = []
        sess.run(tf.global_variables_initializer())
        for i in range(epoch_num):
            err_list.append(1 - sess.run(acc, feed_dict={X: xxData, Y: xxLabel}))
            loss_list.append(sess.run(loss, feed_dict={X: xxData, Y: xxLabel}))

            for j in range(data_num//batch_size):
                Xn = trainData[j*batch_size:(j+1)*batch_size] #make batches by wrapping
                Yn = trainLabel[j*batch_size:(j+1)*batch_size]
                sess.run(train, feed_dict={X: Xn, Y: Yn}) # train the model
            #report percentage progress
            percent_prog = (i*100/epoch_num)
            print(percent_prog,'% completed')
            if visualization==1:
                if percent_prog>=25 and p25==0:
                    W25=sess.run(W1)
                    print('Current classification error: ', err_list[-1])
                    p25=1
                elif percent_prog>=50 and p50==0:
                    #W50=sess.run(W1)
                    print('Current classification error: ', err_list[-1])
                    p50=1
                elif percent_prog>=75 and p75==0:
                    #W75=sess.run(W1)
                    print('Current classification error: ', err_list[-1])
                    p75=1
                elif i == epoch_num-1:
                    print('Final classification error: ', err_list[-1])
                    W100=sess.run(W1)
    #if visualization ==1:
    #    print('dropout visualization fin, dropout=',dropout)
    #    return [W25,W50,W75,W100]

    print('sub-task finished')
    return loss_list,err_list

def Questions(test=0,Q112=0,Q113=0,Q121=0,Q122=0,Q131=0,Q132=1,Q141=0):
    if test==1:
        for eta in [0.005,0.001,0.0001]:
            train_loss, train_err = neural_net(trainData, trainLabel,learn_rate=eta)
            plt.plot(train_loss,label=str(eta)+"loss")
            plt.plot(train_err,label=str(eta)+'error')
            plt.legend(loc='upper right', fontsize='small')
        plt.show()
    if Q112==1:
        train_loss, train_err = neural_net(trainData,trainLabel)
        valid_loss, valid_err = neural_net(validData, validLabel)
        test_loss, test_err = neural_net(testData, testLabel)

        plt.figure()
        plt.subplot(211)
        plt.plot(train_loss, label='training cross-entropy loss')
        plt.plot(valid_loss, label='validation cross-entropy loss')
        plt.plot(test_loss, label='test cross-entropy loss')
        plt.ylabel('cross entropy loss')
        plt.legend(fontsize='small')
        plt.subplot(212)
        plt.plot(train_err, label='training classification error')
        plt.plot(valid_err, label='validation classification error')
        plt.plot(test_err, label='test classification error')
        plt.xlabel('number of epochs')
        plt.ylabel('classification error')
        plt.legend(loc='upper right', fontsize='small')
        plt.savefig('Q112.jpg')
        plt.show()
    if Q113==1:
        epoch_num = 400
        train_loss, train_err = neural_net(trainData, trainLabel,epoch_num=epoch_num)
        valid_loss, valid_err = neural_net(validData, validLabel,epoch_num=epoch_num)
        test_loss, test_err = neural_net(testData, testLabel,epoch_num=epoch_num)

        plt.figure()
        plt.subplot(211)
        plt.plot(train_loss, label='training cross-entropy loss')
        plt.plot(valid_loss, label='validation cross-entropy loss')
        plt.plot(test_loss, label='test cross-entropy loss')
        plt.ylabel('cross entropy loss')
        plt.legend(fontsize='small')
        plt.subplot(212)
        plt.plot(train_err, label='training classification error')
        plt.plot(valid_err, label='validation classification error')
        plt.plot(test_err, label='test classification error')
        plt.xlabel('number of epochs')
        plt.ylabel('classification error')
        plt.legend(loc='upper right', fontsize='small')
        plt.savefig('Q113.jpg')
        plt.show()
    if Q121==1:
        for d_hidden in [100,500,1000]:
            valid_loss,valid_err = neural_net(validData, validLabel,d_hidden=d_hidden)
            plt.plot(valid_err,label=str(d_hidden))
        plt.legend(loc='upper right', fontsize='small')
        plt.savefig('Q121ref.jpg')
        plt.show()
        if 0:
            test_loss, test_err = neural_net(testData, testLabel, d_hidden=None,epoch_num=100)
            print('best test classification error with ____ hidden units is ',np.mean(test_err[-10:]))
    if Q122==1:
        train_loss, train_err = neural_net(trainData, trainLabel, epoch_num=50,d_hidden=500,hidden_layer_num=2)
        valid_loss, valid_err = neural_net(validData, validLabel, epoch_num=50,d_hidden=500,hidden_layer_num=2)
        if 0:
            test_loss, test_err = neural_net(testData, testLabel, epoch_num=50,d_hidden=500,hidden_layer_num=2)
            print('best test classification error with 500 + 500 hidden units is ', np.mean(test_err[-10:]))

        plt.figure()
        plt.plot(train_err, label='training classification error')
        plt.plot(valid_err, label='validation classification error')
        plt.xlabel('number of epochs')
        plt.ylabel('classification error')
        plt.legend(loc='upper right', fontsize='small')
        plt.savefig('Q122.jpg')
        plt.show()
    if Q131==1:
        print('dropout')
        train_loss, train_err = neural_net(trainData, trainLabel, epoch_num=50, d_hidden=1000, hidden_layer_num=1,dropout=1)
        valid_loss, valid_err = neural_net(validData, validLabel, epoch_num=50, d_hidden=1000, hidden_layer_num=1,dropout=1)
        plt.figure()
        plt.plot(train_err, label='training classification error')
        plt.plot(valid_err, label='validation classification error')
        plt.xlabel('number of epochs')
        plt.ylabel('classification error')
        plt.legend(loc='upper right', fontsize='small')
        plt.savefig('131.jpg')
        plt.show()
    if Q132==1:
        print('visualization')
        W = neural_net(trainData, trainLabel, epoch_num=50, d_hidden=1000, hidden_layer_num=1,
                                           dropout=1,visualization=1)
        W0 = neural_net(trainData, trainLabel, epoch_num=50, d_hidden=1000, hidden_layer_num=1,
                        dropout=0, visualization=1)

        for j in range(4):
            plt.figure(figsize=(20,30))
            plt.title('Dropout' + str((j + 1) * 25) + '%')
            for i in range(1000):
                print(i)
                plt.subplot(25,40, i+1)
                plt.axis('off')
                plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                                labeltop='off', labelright='off', labelbottom='off')
                plt.imshow(W[j][:,i].reshape(28, 28), cmap='gray')
            plt.savefig('Q132_dropout_p'+str((j+1)*25)+'.png')

        for j in range(4):
            plt.figure(figsize=(20,30))
            plt.title('No dropout'+str((j+1)*25)+'%')
            for i in range(1000):
                print(i)
                plt.subplot(25,40, i+1)
                plt.axis('off')
                plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                                labeltop='off', labelright='off', labelbottom='off')
                plt.imshow(W0[j][:,i].reshape(28, 28), cmap='gray')
            plt.savefig('Q132_Nodropout_p'+str((j+1)*25)+'.png')
        plt.show()
    if Q141==1:
        for seed in [1000056681,1000056682,1000056683,1000056684,1000056685]:
            np.random.seed(seed)
            learn_rate = np.exp(np.random.rand()*3-7.5)

            hidden_layer_num = np.random.randint(2)+1
            #hidden_layer_num = np.random.randint(5)+1
            d_hidden = np.random.randint(400)+100
            weight_decay = np.exp(np.random.rand()*3-9)
            dropout= np.random.randint(2)

            print('seed',seed,'  learn_rate',learn_rate,'  hidden_layer_num',hidden_layer_num, '  d_hidden',d_hidden,'  weight_decay',weight_decay,'  dropout',dropout)

            valid_loss,valid_err = neural_net(validData, validLabel, learn_rate=learn_rate, weight_decay=weight_decay,
                                               d_hidden=d_hidden, hidden_layer_num=hidden_layer_num, dropout=dropout, visualization=1)
            print('Validation class error for this seed:, ', valid_err[-1])
            
            test_loss, test_err = neural_net(testData, testLabel, learn_rate=learn_rate, weight_decay=weight_decay,
                                              d_hidden=d_hidden, hidden_layer_num=hidden_layer_num, dropout=dropout, visualization=1)
            print('Test class error for this seed:, ', test_err[-1])
Questions(test=0
          ,Q112=0
          ,Q113=0
          ,Q121=0
          ,Q122=0
          ,Q131=0
          ,Q132=0
          ,Q141=1)








