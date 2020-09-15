import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').disabled = True

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# [b'batch_label', b'labels', b'data', b'filenames']

from layer2NN import Layer2NN
np.random.seed(42)

class AutoGrad(Layer2NN):
    def __init__(self):
        super().__init__()
        self.initialize_weights_tf()
        self.Xtest  = tf.convert_to_tensor(self.Xtest , dtype=tf.float32)
        self.Ytest  = tf.convert_to_tensor(self.Ytest , dtype=tf.float32)

    def initialize_weights_tf(self):
        self.w1 = tf.Variable(tf.random.normal([self.n_features, self.hidden_dim], stddev=self.std, dtype=tf.float32))
        self.w2 = tf.Variable(tf.random.normal([self.hidden_dim, self.n_classes], stddev=self.std, dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros([self.hidden_dim], tf.float32))
        self.b2 = tf.Variable(tf.zeros([self.n_classes], tf.float32))

    def forward_propogation(self, X):
        n1 = tf.matmul(X, self.w1) + self.b1
        A = tf.nn.relu(n1)
        Y = tf.matmul(A, self.w2) + self.b2
        return Y

    def Loss_tf(self, Y, T, size):
        return (1./size)*tf.reduce_mean((Y - T)**2) + self.reg * (tf.reduce_mean(self.w2**2) + tf.reduce_mean(self.w1**2))

    def train_autoGrad(self):
        n_batches = len(self.Ytrain) // self.batch_size
        Train_losses = []
        for epoch in range(self.n_epoches):
            train_loss = 0
            train_acc = 0

            for i in range(n_batches):
                X = tf.convert_to_tensor(self.Xtrain[i*self.batch_size:(i+1)*self.batch_size], dtype=tf.float32)
                T = tf.convert_to_tensor(self.Ytrain[i*self.batch_size:(i+1)*self.batch_size], dtype=tf.float32)

                with tf.GradientTape() as tape:
                    Y = self.forward_propogation(X)
                    loss = self.Loss_tf(Y, T, self.batch_size)

                [dw1, dw2, db1, db2] = tape.gradient(loss, [self.w1, self.w2, self.b1, self.b2])
                self.w1.assign_sub(dw1, self.learning_rate)
                self.w2.assign_sub(dw2, self.learning_rate)
                self.b1.assign_sub(db1, self.learning_rate)
                self.b2.assign_sub(db2, self.learning_rate)

                acc = self.evaluation(X, T)
                train_loss += loss.numpy()
                train_acc += acc
                Train_losses.append(train_loss)

            Y = self.forward_propogation(self.Xtest)
            test_loss = self.Loss_tf(Y, self.Ytest, len(self.Ytest)).numpy()
            test_acc =  self.evaluation(self.Xtest, self.Ytest)
            print(" Epoch : {} , Train Loss : {} , Train Accuracy : {} Test Loss : {} Test Accuracy : {}".format(epoch, round(train_loss,5), round(train_acc/n_batches,3), round(test_loss,5), round(test_acc,3)))

    def evaluation(self, X, T):
        Y = self.forward_propogation(X)
        Y = np.argmax(Y.numpy(), axis=-1)
        T = np.argmax(T.numpy(), axis=-1)
        return np.mean(Y==T)

if __name__ == "__main__":
    model = AutoGrad()
    model.train_autoGrad()