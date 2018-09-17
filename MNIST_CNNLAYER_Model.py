import os
from os.path import dirname, realpath

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras import backend as K
import matplotlib.pyplot as plt


class Cvml:
    def __init__(self):
        self.img_rows = 28        
        self.img_cols = 28        
        self.plots_dir = "resulting_plots"

    def create_and_train_model(self,
                               batch_size=256,
                               num_classes=10, 
                               epochs=15, 
                               layers=2, 
                               activation_fn='relu', 
                               add_dropout=False,
                               kernel_size=(3, 3), 
                               no_of_kernels=32,
                               l2_reg=0.0):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
        input_shape = (self.img_rows, self.img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('X_TRAIN SHAPE:', x_train.shape)
        print('NO OF TRAINING EXAMPLES', x_train.shape[0],)
        print('NO OF TESTS EXAMPLES', x_test.shape[0],)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = Sequential()
        model.add(Conv2D(no_of_kernels, 
                         kernel_size=kernel_size,
                         input_shape=input_shape, 
                         kernel_regularizer=l2(l2_reg)))
        model.add(Activation(activation_fn))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        for i in range(1,layers) :
            model.add(Conv2D(64, kernel_size, kernel_regularizer=l2(l2_reg)))
            model.add(Activation(activation_fn))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            if add_dropout:
                model.add(Dropout(0.25))
        model.add(Flatten())

        model.add(Dense(128, activation=activation_fn))
        model.add(Activation(activation_fn))
        if add_dropout:
            model.add(Dropout(0.25))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('LOSS IN TEST DATA:', score[0])
        print('ACCURACY IN TEST DATA:', score[1])
        return history, score

    def test_no_of_layers(self, plot_type='layers'):
        h1, s1 = self.create_and_train_model(layers=1)
        h2, s2 = self.create_and_train_model(layers=2)
        h3, s3 = self.create_and_train_model(layers=3)

        legend1 = '1 Convolution layer(Test accuracy - {})'.format(s1[1])
        legend2 = '2 Convolution layer(Test accuracy - {})'.format(s2[1])
        legend3 = '3 Convolution layers(Test accuracy - {})'.format(s3[1])

        self.plot([h1.history['val_acc'],h2.history['val_acc'],h3.history['val_acc']],
                  "MODEL ACCURACY BASED ON NO. OF HIDDEN LAYERS",
                  "EPOCH",
                  "ACCURACY",
                  [legend1, legend2, legend3],
                  self.plots_dir + '/accuracy_cnn_{}.png'.format(plot_type)
                  )

    def test_activation_fun(self, plot_type='activation'):
        h1, s1 = self.create_and_train_model(activation_fn='relu')
        h2, s2 = self.create_and_train_model(activation_fn='sigmoid')
        h3, s3 = self.create_and_train_model(activation_fn='tanh')
        h4, s4 = self.create_and_train_model(activation_fn='elu')

        legend1 = 'ReLU(Test accuracy - {})'.format(s1[1])
        legend2 = 'Sigmoid(Test accuracy - {})'.format(s2[1])
        legend3 = 'Tanh(Test accuracy - {})'.format(s3[1])
        legend4 = 'elu(Test accuracy - {})'.format(s4[1])

        self.plot([h1.history['val_acc'],h2.history['val_acc'],h3.history['val_acc'],h4.history['val_acc']],
                  "MODEL ACCURACY BASED ON TYPE OF ACTIVATION FUNCTION",
                  "EPOCH",
                  "ACCURACY",
                  [legend1, legend2, legend3, legend4],
                  self.plots_dir + '/accuracy_cnn_{}.png'.format(plot_type)
                  )

    def test_kernel_size(self, plot_type='kernel_size'):
        h1, s1 = self.create_and_train_model(kernel_size=(3, 3))
        h2, s2 = self.create_and_train_model(kernel_size=(5, 5))
        h3, s3 = self.create_and_train_model(kernel_size=(7, 7))
        h4, s4 = self.create_and_train_model(kernel_size=(9, 9))

        legend1 = '3x3 kernel(Test accuracy - {})'.format(s1[1])
        legend2 = '5x5 kernel(Test accuracy - {})'.format(s2[1])
        legend3 = '7x7 kernel(Test accuracy - {})'.format(s3[1])
        legend4 = '9x9 kernel(Test accuracy - {})'.format(s4[1])
        self.plot([h1.history['val_acc'],h2.history['val_acc'],h3.history['val_acc'],h4.history['val_acc']],
                  "MODEL ACCURACY BASED ON KERNEL SIZE",
                  "EPOCH",
                  "ACCURACY",
                  [legend1, legend2, legend3, legend4],
                  self.plots_dir + '/accuracy_cnn_{}.png'.format(plot_type)
                  )

    def test_no_of_kernels(self, plot_type = 'no_of_kernels'):
        h1,s1 = self.create_and_train_model(no_of_kernels=16)
        h2,s2 = self.create_and_train_model(no_of_kernels=24)
        h3,s3 = self.create_and_train_model(no_of_kernels=32)
        h4,s4 = self.create_and_train_model(no_of_kernels=64)

        legend1 = '16 kernels(Test accuracy - {})'.format(s1[1])
        legend2 = '24 kernels(Test accuracy - {})'.format(s2[1])
        legend3 = '32 kernels(Test accuracy - {})'.format(s3[1])
        legend4 = '64 kernels(Test accuracy - {})'.format(s4[1])
        self.plot([h1.history['val_acc'],h2.history['val_acc'],h3.history['val_acc'],h4.history['val_acc']],
                  "MODEL ACCURACY BASED ON NO OF KERNELS",
                  "EPOCH",
                  "ACCURACY",
                  [legend1, legend2, legend3, legend4],
                  self.plots_dir + '/accuracy_cnn_{}.png'.format(plot_type)
                  )

    def test_overfitting(self, plot_type='overfitting'):
        h1, s1 = self.create_and_train_model(epochs=50)
        h2, s2 = self.create_and_train_model(epochs=50, add_dropout=True)
        h3, s3 = self.create_and_train_model(epochs=50, l2_reg=0.01)
        h4, s4 = self.create_and_train_model(epochs=50, l2_reg=0.01,add_dropout=True)

        self.plot([h1.history['acc'], h1.history['val_acc']],
                  'MODEL ACCURACY',
                  'EPOCH',
                  'ACCURACY',
                  ['train', 'test'],
                  self.plots_dir + '/accuracy_cnn_test_train.png'.format(plot_type)
                  )
        self.plot([h1.history['loss'], h1.history['val_loss']],
                  'MODEL LOSS',
                  'EPOCH',
                  'LOSS',
                  ['train', 'test'],
                  self.plots_dir + '/loss_cnn_test_train.png'.format(plot_type)
                  )

        legend1 = 'No technique used(Test accuracy - {})'.format(s1[1])
        legend2 = 'Dropout(Test accuracy - {})'.format(s2[1])
        legend3 = 'L2 regularization(Test accuracy - {})'.format(s3[1])
        legend4 = 'L2 + Dropout (Test accuracy - {})'.format(s4[1])
        self.plot([h1.history['loss'],h2.history['loss'],h3.history['loss'], h4.history['loss']],
                  'MODEL LOSS USING DIFFERENT OVERFITTING TECHNIQUES',
                  'EPOCH',
                  'ACCURACY',
                  [legend1, legend2, legend3, legend4],
                  self.plots_dir + '/loss_cnn_{}.png'.format(plot_type)
                  )

    def plot(self, plots, title, xlabel, ylabel, legend, save_path):
        for i in plots:
            plt.plot(i)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legend, loc="lower right")
        plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    obj = Cvml()
    obj.test_no_of_layers()
    obj.test_activation_fun()
    obj.test_kernel_size()
    obj.test_no_of_kernels()
    obj.test_overfitting()
    pass
