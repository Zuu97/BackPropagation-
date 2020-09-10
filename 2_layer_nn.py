import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').disabled = True

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import tensorflow as tf
# [b'batch_label', b'labels', b'data', b'filenames']

class Layer2NN(object):
    def __init__(self):
        self.train_batch_prefix = 'data_batch_'
        self.test_batch = 'test_batch'
        self.n_batches = 5
        self.train_size = 10000
        self.test_size = 10000
        self.batch_size = 128
        self.learning_rate = 1.4e-4
        self.std=1e-4
        self.reg=5e-5
        self.learning_rate_decay=0.999
        self.hidden_dim = 300
        self.n_epoches = 30
        self.Ytrain, self.Xtrain = self.load_train_data()
        self.Ytest , self.Xtest  = self.load_test_data()
        self.n_features = self.Xtrain.shape[1]
        self.n_classes = len(set(self.Ytrain))

        self.Ytrain =  tf.keras.utils.to_categorical(self.Ytrain, num_classes=self.n_classes)
        self.Ytest  = tf.keras.utils.to_categorical(self.Ytest, num_classes=self.n_classes)

        print(" Xtrain Shape : {} ".format(self.Xtrain.shape))
        print(" Ytrain Shape : {} ".format(self.Ytrain.shape))
        print(" Xtest  Shape : {} ".format(self.Xtest.shape))
        print(" Ytest  Shape : {} \n".format(self.Ytest.shape))

    def initialize_weights(self):
        self.w1 = self.std*np.random.randn(self.n_features, self.hidden_dim)
        self.w2 = self.std*np.random.randn(self.hidden_dim, self.n_classes)
        self.b1 = np.zeros(self.hidden_dim)
        self.b2 = np.zeros(self.n_classes)

    @staticmethod
    def extract_batch_data(file):
        with open(file, 'rb') as data:
            data_dict = pickle.load(data, encoding='bytes')
        return data_dict

    def load_train_data(self):
        train_labels = []
        train_images = []
        for i in range(1, self.n_batches + 1):
            batch_file = self.train_batch_prefix + str(i)
            data_dict = Layer2NN.extract_batch_data(batch_file)
            train_labels.extend(data_dict[b'labels'])
            train_images.extend(data_dict[b'data'])
        all_train_labels, all_train_images = np.array(train_labels), np.array(train_images)
        all_train_labels, all_train_images = shuffle(all_train_labels, all_train_images)

        N = len(all_train_labels)
        idx = np.random.choice(N, self.train_size)

        train_labels = all_train_labels[idx]
        train_images = all_train_images[idx]

        return train_labels, train_images

    def load_test_data(self):
        data_dict = Layer2NN.extract_batch_data(self.test_batch)
        test_images = data_dict[b'data']
        test_labels = data_dict[b'labels']
        all_test_labels, all_test_images = np.array(test_labels), np.array(test_images)
        all_test_labels, all_test_images = shuffle(all_test_labels, all_test_images)

        N = len(all_test_labels)
        idx = np.random.choice(N, self.test_size)

        test_labels = all_test_labels[idx]
        test_images = all_test_images[idx]

        return test_labels, test_images

    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    @staticmethod
    def relu(z):
        return z * (z > 0)

    @staticmethod
    def softmax(z):
        return np.exp(z) / (np.sum(np.exp(z), axis=-1))

    @staticmethod
    def derivative_of_sigmoid(z):
        return Layer2NN.sigmoid(z) - (Layer2NN.sigmoid(z) ** 2)

    @staticmethod
    def derivative_of_relu(z):
        return (z > 0)

    def forward_propogation(self, X):
        n1 = X.dot(self.w1) + self.b1
        A = Layer2NN.relu(n1)

        Y = A.dot(self.w2) + self.b2
        return Y, A

    def Loss(self, Y, T):
        return 1./self.batch_size*np.square(Y - T).sum() + self.reg * (np.sum(self.w2 * self.w2) + np.sum( self.w1 * self.w1 ))

    def backward_propogation(self, X, Y, A, T):
        dY = 1./self.batch_size*2.0*(Y - T)
        self.dw2 = A.T.dot(dY) + self.reg*self.w2
        self.db2 = dY.sum(axis=0)
        dA = dY.dot(self.w2.T)
        self.dw1 = X.T.dot(dA*Layer2NN.derivative_of_relu(A)) + self.reg*self.w1
        self.db1 = (dA*Layer2NN.derivative_of_relu(A)).sum(axis=0)

    def gradient_descent(self):
        self.w2 = self.w2 - self.learning_rate * self.dw2
        self.w1 = self.w1 - self.learning_rate * self.dw1
        self.b2 = self.b2 - self.learning_rate * self.db2
        self.b1 = self.b1 - self.learning_rate * self.db1
        self.learning_rate = self.learning_rate* self.learning_rate_decay

    def evaluation(self, Y, T):
        Y = np.argmax(Y, axis=-1)
        T = np.argmax(T, axis=-1)
        return np.mean(Y==T)

    def train_model(self):
        self.initialize_weights()
        n_batches = len(self.Ytrain) // self.batch_size
        Train_losses = []
        for epoch in range(self.n_epoches):
            train_loss = 0
            train_acc = 0
            for i in range(n_batches):
                X = self.Xtrain[i*self.batch_size:(i+1)*self.batch_size]
                T = self.Ytrain[i*self.batch_size:(i+1)*self.batch_size]

                Y, A = self.forward_propogation(X)
                loss = self.Loss(Y, T)
                acc = self.evaluation(Y, T)
                train_loss += loss
                train_acc += acc

                self.backward_propogation(X, Y, A, T)
                self.gradient_descent()

                Train_losses.append(train_loss)
            print(" Epoch : {} , Loss : {} , Accuracy : {} ".format(epoch, round(train_loss,3), round(train_acc/n_batches,3)))


if __name__ == "__main__":
    model = Layer2NN()
    model.train_model()