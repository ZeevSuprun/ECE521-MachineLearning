#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np


#######################################

def distance(matrixX, matrixZ):
    (N1, d) = matrixX.shape
    (N2, d) = matrixZ.shape

    #Reshape matrix Z into being a 1x(N2*d) row.
    matrixZreshaped = tf.reshape(matrixZ, [-1, N2 * d])
    #tile matrixX so that it is repeated N2 times horizontally. dim N1 x (N2*d)
    repMatrixX = tf.tile(matrixX, [1, N2])
    #tile the reshaped matrixZ so that it is repeated for N1 rows. dim N1 x (N2*d)
    repMatrixZ = tf.tile(matrixZreshaped, [N1, 1])


    differenceMatrix = tf.subtract(repMatrixZ, repMatrixX)
    print('differenceMatrix Size = ',differenceMatrix.eval(session=sess).size)
    differenceVector = tf.reshape(differenceMatrix, [-1, d])
    print('differenceVector Size = ',differenceMatrix.eval(session=sess).size)
    #elementwise multiply
    matrixMul = tf.multiply(differenceVector, differenceVector)
    print('MatrixMulSize = ',matrixMul.eval(session=sess).size)
    distanceMatrix = tf.reshape(tf.reduce_sum(matrixMul, 1), [N1, N2])
    # distanceMatrix contains pairwise Euclidean distances

    return distanceMatrix



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
