import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout ,Activation
from keras.optimizers import RMSprop
from keras.regularizers import l2
import matplotlib.pyplot as plt


class Cvml:
    def __init__(self):
        self.plots_dir = "resulting_plots2"

    def create_and_train_model(self,
                               batch_size=256,
                               num_classes=10,
                               epochs=25,
                               layers=2,
                               activation_fn='relu',
                               add_dropout=False,
                               l2_reg=0.0):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('NO OF TRAINING EXAMPLES', x_train.shape[0],)
        print('NO OF TESTS EXAMPLES', x_test.shape[0],)

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = Sequential()
        model.add(Dense(100, kernel_regularizer=l2(l2_reg), input_shape=(784,)))
        model.add(Activation(activation_fn))
        if add_dropout:
            model.add(Dropout(0.25))

        if layers >= 2:
            model.add(Dense(50, kernel_regularizer=l2(l2_reg)))
            model.add(Activation(activation_fn))
            if add_dropout:
                model.add(Dropout(0.25))
        if layers >= 3:
            model.add(Dense(30, kernel_regularizer=l2(l2_reg)))
            model.add(Activation(activation_fn))
            if add_dropout:
                model.add(Dropout(0.25))

        model.add(Dense(10, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer='adagrad',
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

    def test_no_of_layers(self,
                          plot_type='layers'):
        h1, s1 = self.create_and_train_model(layers=1)
        h2, s2 = self.create_and_train_model(layers=2)
        h3, s3 = self.create_and_train_model(layers=3)

        legend1 = '1 layer(Test accuracy - {})'.format(s1[1])
        legend2 = '2 layers(Test accuracy - {})'.format(s2[1])
        legend3 = '3 layers(Test accuracy - {})'.format(s3[1])
        self.plot([h1.history['val_acc'],h2.history['val_acc'],h3.history['val_acc']],
                  'MODEL ACCURACY BASED ON NUMBER OF HIDDEN LAYERS',
                  'EPCOHS',
                  'ACCURACY',
                  [legend1, legend2, legend3],
                  self.plots_dir + '/accuracy_simple_{}.png'.format(plot_type)
                  )

    def test_epochs(self,
                    plot_type='epochs'):
        h1, s1 = self.create_and_train_model(epochs=100)
        legend1 = 'ACCURACY'
        legend2 = 'LOSS'

        self.plot(
            [h1.history['val_acc'], h1.history['val_loss']],
            'MODEL ACCURACY AND LOSS',
            'EPCOHS',
            'ACCURACY',
            [legend1, legend2],
            self.plots_dir + '/accuracy_simple_{}.png'.format(plot_type)
            )

    def test_activation_fun(self,
                            plot_type='activation'):
        h1, s1 = self.create_and_train_model(activation_fn='relu')
        h2, s2 = self.create_and_train_model(activation_fn='sigmoid')
        h3, s3 = self.create_and_train_model(activation_fn='tanh')
        h4, s4 = self.create_and_train_model(activation_fn='elu')

        legend1 = 'ReLU(Test accuracy - {})'.format(s1[1])
        legend2 = 'Sigmoid(Test accuracy - {})'.format(s2[1])
        legend3 = 'Tanh(Test accuracy - {})'.format(s3[1])
        legend4 = 'elu(Test accuracy - {})'.format(s4[1])

        self.plot(
            [h1.history['val_acc'], h2.history['val_acc'], h3.history['val_acc'], h4.history['val_acc']],
            'MODEL ACCURACY BASED ON TYPE OF ACTIVATION FUNCTION',
            'EPCOHS',
            'ACCURACY',
            [legend1, legend2, legend3, legend4],
            self.plots_dir + '/accuracy_simple_{}.png'.format(plot_type)
            )

    def test_overfitting(self,
                         plot_type='overfitting'):
        h1, s1 = self.create_and_train_model(epochs=50)
   5     h2, s2 = self.create_and_train_model(epochs=50, add5_dropout=True)
        h3, s3 = self.create_and_train_model(epochs=50, l2_5reg=0.0001)
        h4, s4 = self.create_and_train_model(epochs=50, add_dropout=True,
                                             l2_reg=0.0001)

        self.plot(
            [h1.history['acc'], h1.history['val_acc']],
            'MODEL ACCURACY ',
            'EPCOHS',
            'ACCURACY',
            ['train', 'test'],
            self.plots_dir + '/accuracy_simple_test_train.png'.format(plot_type)
        )

        self.plot(
            [h1.history['loss'], h1.history['val_loss']],
            'MODEL LOSS ',
            'EPCOHS',
            'LOSS',
            ['train', 'test'],
            self.plots_dir + '/loss_simple_test_train.png'.format(plot_type)
        )

        legend1 = 'NO TECHNIQUE USED(Test accuracy - {})'.format(s1[1])
        legend2 = 'DROPOUT(Test accuracy - {})'.format(s2[1])
        legend3 = 'L2 REGULARIZATION(Test accuracy - {})'.format(s3[1])
        legend4 = 'L2 REGULARIZATION with DROPOUT(Test accuracy - {})'.format(
            s4[1])

        self.plot(
            [h1.history['val_loss'], h2.history['val_loss'], h3.history['loss'], h4.history['val_loss']],
            'MODEL LOSS USING DIFFERENT OVERFITTING TECHNIQUES',
            'EPCOHS',
            'LOSS',
            [legend1, legend2, legend3, legend4],
            self.plots_dir + '/loss_simple_{}.png'.format(plot_type)
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
    obj.test_epochs()
    obj.test_activation_fun()
    obj.test_overfitting()
