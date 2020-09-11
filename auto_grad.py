import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').disabled = True

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
# [b'batch_label', b'labels', b'data', b'filenames']

from layer2NN import Layer2NN
np.random.seed(42)

class AutoGrad(Layer2NN):
    def __init__(self):
        super().__init__()
        self.initialize_weights()
        self.w1 = tf.convert_to_tensor(self.w1)
        self.w2 = tf.convert_to_tensor(self.w2)
        self.b1 = tf.convert_to_tensor(self.b1)
        self.b2 = tf.convert_to_tensor(self.b2)


    def train_autoGrad(self):
        n_batches = len(self.Ytrain) // self.batch_size
        Train_losses = []
        for epoch in range(self.n_epoches):
            train_loss = 0
            train_acc = 0

            for i in range(n_batches):
                X = self.Xtrain[i*self.batch_size:(i+1)*self.batch_size]
                T = self.Ytrain[i*self.batch_size:(i+1)*self.batch_size]

                Y, A = self.forward_propogation(X)

                with tf.GradientTape as tape:
                    Y, _ = self.forward_propogation(X)
                    loss = self.Loss(Y, T)

                grads = tape.gradient(loss, [self.w1, self.w2, self.b1, self.b2])
                self.w1.assign_sub(grads[0], self.learning_rate)
                self.w2.assign_sub(grads[1], self.learning_rate)
                self.b1.assign_sub(grads[2], self.learning_rate)
                self.b2.assign_sub(grads[3], self.learning_rate)