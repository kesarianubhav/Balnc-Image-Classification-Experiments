from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Dense, Flatten
import numpy as np
import pandas as pd
from time import time
import keras


class Classifier(object):

    def __init__(self, batch_size=256, n_epochs=20, n_classes=4):
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self._trained = False
        self.batch_size = batch_size

    def create_architecture(self, input_shape, output_dimension, model='None'):
        self.input_shape = input_shape
        self.output_dimension = output_dimension
        opt = keras.optimizers.RMSprop(lr=0.00007)
#         global self.classifier
        if model == 'None':
            self.classifier = Sequential()
            self.classifier.add(Conv2D(
                32, 3, 3, input_shape=(input_shape), activation='relu'))
            self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
            self.classifier.add(Flatten())
            self.classifier.add(
                Dense(output_dim=output_dimension, activation='sigmoid'))
            self.classifier.compile(
                optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        # else:
        #     self.classifier = VGG16()
        #     image = preprocess_input(image)

    def train_model(self, input_tensors, output_tensors):
        print("Training started\n")
        # self.__create_model()
        if self._trained == False:
            x_train = input_tensors
            x_train = np.array(x_train).reshape((-1,
                                                 self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            y_train = output_tensors
            y_train = np.array(y_train).reshape((-1, self.n_classes))
            # plot_losses = PlotLearning()
            self.hist = self.classifier.fit(x_train, y_train,
                                            batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False, validation_split=0.1)

            self._trained = True
            t = time()
#             self.model.save("saved_model/" + str(t))

    def test(self, X_test, Y_test):
        return

    def predict(self, input_tensors):
        outputs = []

        for i in input_tensors:
            i = np.array(i).reshape((-1,
                                     self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            outputs.append(self.classifier.predict_classes(i))

        return outputs
