#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def findResponsibilityVector(distanceMatrix, k):
    distanceVector = tf.reshape(distanceMatrix, [1, tf.size(distanceMatrix)])
    topIndices = tf.nn.top_k(distanceVector, k).indices
    responsibilityVector = tf.matrix_transpose(tf.zeros_like(distanceVector)).eval(session=sess)
    responsibilityVector[topIndices.eval(session=sess)] = tf.divide(1, k)
    responsibilityMatrix = tf.reshape(responsibilityVector, tf.shape(distanceMatrix))
    return tf.convert_to_tensor(responsibilityMatrix)
    
    
############## TEST DATA ##############
k = 4
distanceMatrix = tf.constant([11.0, 0.0, 3.0, 4.0, 5.0, 2.0, 1.0, 9.0, 3.0, 1.0, 7.0, 6.0], shape = [3, 4])
#######################################

responsibilityMatrix = findResponsibilityVector(distanceMatrix, k)

eval0 = distanceMatrix.eval(session=sess)
eval1 = responsibilityMatrix.eval(session=sess)

