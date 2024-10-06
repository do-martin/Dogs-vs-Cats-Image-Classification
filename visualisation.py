import os
import random
import PIL
import PIL.Image as Image
import numpy as np
import pandas as pd
import seaborn as sns
from keras.src.utils import load_img
from matplotlib import pyplot as plt


class Visualisation:
    def __init__(self):
        self.input_path = []
        self.label = []
        self.df = pd.DataFrame()

    def add_label_by_path(self, main_path):
        for class_name in os.listdir(main_path):
            for path in os.listdir(main_path + '\\' + class_name):
                if class_name == 'cats':
                    self.label.append(0)
                else:
                    self.label.append(1)
                self.input_path.append(
                    os.path.join(main_path, class_name, path))
        print(self.input_path[0], self.label[0])

    def get_modified_dataframe(self):
        self.df['images'] = self.input_path
        self.df['label'] = self.label
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.head()
        print(len(self.df))
        return self.df

    def validate_data(self):
        for i in self.df['images']:
            if '.jpg' not in i:
                print(i)

        l = []
        for image in self.df['images']:
            try:
                img = PIL.Image.open(image)
            except:
                l.append(image)
        print(l)

    '''"# delete db files
    df = df[df['images'] != 'PetImages/Dog/Thumbs.db']
    df = df[df['images'] != 'PetImages/Cat/Thumbs.db']
    df = df[df['images'] != 'PetImages/Cat/666.jpg']
    df = df[df['images'] != 'PetImages/Dog/11702.jpg']
    len(df)'''

    def display_images_of_dogs_or_cats(self, animal_with_uppercase):
        plt.figure(figsize=(25, 25))
        temp = ''
        if animal_with_uppercase == 'Dogs':
            temp = self.df[self.df['label'] == 1]['images']
        else:
            temp = self.df[self.df['label'] == 0]['images']
        start = random.randint(0, len(temp))
        files = temp[start:start + 25]
        for index, file in enumerate(files):
            plt.subplot(5, 5, index + 1)
            img = load_img(file)
            img = np.array(img)
            plt.imshow(img)
            plt.title(animal_with_uppercase)
            plt.axis('off')

    def count_plot(self):
        sns.countplot(self.df['label'])

    @staticmethod
    def show_training_history(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
        plt.title('Accuracy Graph')
        plt.legend()
        plt.figure()

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title('Loss Graph')
        plt.legend()
        plt.show()
