import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').disabled = True

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
# [b'batch_label', b'labels', b'data', b'filenames']

np.random.seed(42)

class Layer2NN(object):
    def __init__(self):
        self.train_batch_prefix = 'data/data_batch_'
        self.test_batch = 'data/test_batch'
        self.n_batches = 5
        self.train_size = 50000
        self.test_size = 10000
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.std=1e-6
        self.reg=5e-6
        self.learning_rate_decay=0.999
        self.hidden_dim = 300
        self.n_epoches = 30
        self.loss_img = 'visualization/loss_variation_{}.png'
        self.acc_img = 'visualization/acc_variation_{}.png'
        self.Ytrain, self.Xtrain = self.load_train_data()
        self.Ytest , self.Xtest  = self.load_test_data()
        self.n_features = self.Xtrain.shape[1]
        self.n_classes = len(set(self.Ytrain))

        print(" Xtrain Shape : {} ".format(self.Xtrain.shape))
        print(" Ytrain Shape : {} ".format(self.Ytrain.shape))
        print(" Xtest  Shape : {} ".format(self.Xtest.shape))
        print(" Ytest  Shape : {} \n".format(self.Ytest.shape))

        print(" Batch size : {}".format(self.batch_size))
        print(" learning rate : {}".format(self.learning_rate))

        self.Ytrain =  tf.keras.utils.to_categorical(self.Ytrain, num_classes=self.n_classes)
        self.Ytest  = tf.keras.utils.to_categorical(self.Ytest, num_classes=self.n_classes)

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
        return (1./len(Y))*np.square(Y - T).sum() + self.reg * (np.sum(self.w2 * self.w2) + np.sum( self.w1 * self.w1 ))

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
        # self.learning_rate = self.learning_rate* self.learning_rate_decay

    def evaluation(self, X, T):
        Y, A = self.forward_propogation(X)
        Y = np.argmax(Y, axis=-1)
        T = np.argmax(T, axis=-1)
        return np.mean(Y==T)

    def train_model(self):
        self.initialize_weights()
        n_batches = len(self.Ytrain) // self.batch_size
        Train_losses = []
        Train_accuracy = []
        Test_losses = []
        Test_accuracy = []

        for epoch in range(self.n_epoches):
            train_loss = 0
            train_acc = 0

            for i in range(n_batches):
                X = self.Xtrain[i*self.batch_size:(i+1)*self.batch_size]
                T = self.Ytrain[i*self.batch_size:(i+1)*self.batch_size]

                Y, A = self.forward_propogation(X)
                loss = self.Loss(Y, T)
                acc = self.evaluation(X, T)
                train_loss += loss
                train_acc += acc

                self.backward_propogation(X, Y, A, T)
                self.gradient_descent()

            Y, A = self.forward_propogation(self.Xtest)
            test_loss = self.Loss(Y, self.Ytest)
            test_acc =  self.evaluation(self.Xtest, self.Ytest)


            Train_losses.append(train_loss/n_batches)
            Test_losses.append(test_loss)
            Train_accuracy.append(train_acc/n_batches)
            Test_accuracy.append(test_acc)
            print(" Epoch : {} , Train Loss : {} , Train Accuracy : {} Test Loss : {} Test Accuracy : {}".format(epoch, round(train_loss/n_batches,3), round(train_acc/n_batches,3), round(test_loss,3), round(test_acc,3)))

        self.visualize_cum_metrices(Train_losses,Test_losses,  Train_accuracy,Test_accuracy)

    def visualize_cum_metrices(self, Train_losses,Test_losses,  Train_accuracy,Test_accuracy):
        Train_losses,Test_losses = np.array(Train_losses),np.array(Test_losses)
        cum_train_loss = np.cumsum(Train_losses) / np.arange(1,self.n_epoches+1)
        cum_test_loss = np.cumsum(Test_losses) / np.arange(1,self.n_epoches+1)

        plt.plot(cum_train_loss, 'r', label='Training Loss')
        plt.plot(cum_test_loss, 'b', label='validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(self.loss_img.format('constant_lr_1e-3'))
        plt.legend()
        plt.show()

        Train_accuracy,Test_accuracy = np.array(Train_accuracy),np.array(Test_accuracy)
        cum_train_accuracy = np.cumsum(Train_accuracy) / np.arange(1,self.n_epoches+1)
        cum_test_accuracy = np.cumsum(Test_accuracy) / np.arange(1,self.n_epoches+1)

        plt.plot(cum_train_accuracy, 'r', label='Training Accuracy')
        plt.plot(cum_test_accuracy, 'b', label='validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(self.acc_img.format('constant_lr_1e-3'))
        plt.legend()
        plt.show()

    def visualize_metrics(self, Train_losses,Test_losses,  Train_accuracy,Test_accuracy):
        Train_losses,Test_losses = np.array(Train_losses),np.array(Test_losses)

        plt.plot(Train_losses, 'r', label='Training Loss')
        plt.plot(Test_losses, 'b', label='Test Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(self.loss_img.format('constant_lr'))
        plt.legend()
        plt.show()

        Train_accuracy,Test_accuracy = np.array(Train_accuracy),np.array(Test_accuracy)

        plt.plot(Train_accuracy, 'r', label='Training Accuracy')
        plt.plot(Test_accuracy, 'b', label='Test Accuracy')
        plt.title('Training and Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(self.acc_img.format('constant_lr_1e-3'))
        plt.legend()
        plt.show()

if __name__ == "__main__":
    model = Layer2NN()
    model.train_model()