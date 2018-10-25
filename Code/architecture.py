from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Dense, Flatten
import numpy as np
import pandas as pd


class Classifier(object):

    def __init__(self, n_epochs=10, n_classes=4):
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self._trained = False

    def create_architecture(self, input_shape, output_dimension):
        self.input_shape = input_shape
        self.output_dimension = output_dimension
        self.classifier = Sequential()
        self.classifier.add(Conv2D(
            32, 3, 3, input_shape=(input_shape), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Flatten())
        self.classifier.add(
            Dense(output_dim=output_dimension, activation='sigmoid'))
        self.classifier.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
                                            batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False)

            self._trained = True
            t = time()
            self.model.save("saved_model/" + str(t))

    def test(self, X_test, Y_test):
        return

    def predict(self, X_predict):
        outputs = []

        for i in input_tensors:
            i = np.array(i).reshape((-1,
                                     self.n_time_steps, self.n_inputs))
            outputs.append(self.model.predict_classes(i))

        return outputs
