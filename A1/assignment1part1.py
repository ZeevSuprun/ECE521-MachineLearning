#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np


#######################################

def distance(X,Z):
    '''
    Euclidean distance
    Input: X is a N1xd matrix
           Z is a N2xd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||X[i,:]-Z[j,:]||^2
    '''
    #column vector of x norms
    X_norm = tf.reshape(tf.reduce_sum(X**2, axis=1), [X.shape[0],1])
    #row vector of z norms
    Z_norm = tf.reshape(tf.reduce_sum(Z**2, axis=1), [1,Z.shape[0]])
    dist = X_norm + Z_norm-2*tf.matmul(X,tf.transpose(Z))
    return dist



if __name__ == '__main__':
    ############## TEST DATA ##############
    #N1 = 5
    #N2 = 3
    #d = 4
    matX = tf.constant([1.0, 0.0, 3.0, 4.0, 5.0, 2.0, 1.0, 9.0, 0.0, 3.0, 1.0, 7.0, 6.0, 2.0, 0.0, 1.0, 6.0, 1.0, 3.0, 0.0], shape = [5, 4])
    matZ = tf.constant([8.0, 3.0, 0.0, 2.0, 1.0, 5.0, 1.0, 5.0, 2.0, 2.0, 9.0, 0.0], shape = [3, 4])
    matZ2 = tf.constant([8.0, 3.0, 0.0, 2.0], shape = [1, 4])
    #matZerr = tf.constant([8.0, 3.0, 0.0, 2.0]) #using this as an input causes an error.
    zeroMat = tf.zeros([3,4])
    #print(matX.shape)
    #print(matZ.shape)
    #print(zeroMat.shape)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)


    distanceMatrix = distance(matX, matZ2)

    #print(sess.run([matrixX, matrixZ,distanceMatrix]))
    print(sess.run([matX]))
    print(sess.run([matZ]))
    print(sess.run([distanceMatrix]))
    #Note: can also print values by printing variable.eval(session=sess)
