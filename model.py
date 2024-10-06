from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
import time


class Model:
    def __init__(self, train_iterator, val_iterator):
        self.model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.history = None
        self.NAME = "Cats-vs-dog-cnn-64x2-beginning-15-epochs-{}".format(int(time.time()))
        self.tensorboard = TensorBoard(log_dir='logs/{}'.format(self.NAME))

    def create_model_or_continue_training(self, number_epochs):
        self.history = self.model.fit(self.train_iterator, epochs=number_epochs,
                                      validation_data=self.val_iterator,
                                      callbacks=[self.tensorboard])  # normally 10 epochs
        return self.history

    def save_model(self, path):
        self.model.save(path)
        return self.model

    def load_model(self, path):
        self.model = load_model(path)
        return self.model
