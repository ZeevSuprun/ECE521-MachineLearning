#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

############## TEST DATA ##############
N1 = 5
N2 = 3
d = 4
matrixX = tf.constant([1.0, 0.0, 3.0, 4.0, 5.0, 2.0, 1.0, 9.0, 0.0, 3.0, 1.0, 7.0, 6.0, 2.0, 0.0, 1.0, 6.0, 1.0, 3.0, 0.0], shape = [N1, d])
matrixZ = tf.constant([8.0, 3.0, 0.0, 2.0, 1.0, 5.0, 1.0, 5.0, 2.0, 2.0, 9.0, 0.0], shape = [N2, d])
#######################################

matrixZreshaped = tf.reshape(matrixZ, [-1, N2 * d])
repMatrixX = tf.tile(matrixX, [1, N2])
repMatrixZ = tf.tile(matrixZreshaped, [N1, 1])

differenceMatrix = tf.subtract(repMatrixZ, repMatrixX)
differenceVector = tf.reshape(differenceMatrix, [-1, d])
matrixMul = tf.multiply(differenceVector, differenceVector)
distanceMatrix = tf.reshape(tf.reduce_sum(matrixMul, 1), [N1, N2])

eval1 = repMatrixX.eval(session=sess)
eval2 = repMatrixZ.eval(session=sess)
eval3 = differenceMatrix.eval(session=sess)
eval4 = differenceVector.eval(session=sess)
eval5 = matrixMul.eval(session=sess)
eval6 = matrixZreshaped.eval(session=sess)
eval7 = distanceMatrix.eval(session=sess)

# distanceMatrix contains pairwise Euclidean distances 

