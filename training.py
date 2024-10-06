from keras import Model
import keras.preprocessing.image as img
from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class Training:
    def __init__(self, df):
        self.df = df
        self.df['label'] = self.df['label'].astype('str')
        self.df.head()

    def get_iterators(self):
        train, test = train_test_split(self.df, test_size=0.2, random_state=42)
        train_generator = img.ImageDataGenerator(
            # normalization of images
            rescale=1. / 255,
            # augmentation of images to avoid overfitting
            rotation_range=40,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_generator = ImageDataGenerator(rescale=1. / 255)

        train_iterator = train_generator.flow_from_dataframe(
            train,
            x_col='images',
            y_col='label',
            target_size=(128, 128),
            batch_size=512,
            class_mode='binary'
        )

        val_iterator = val_generator.flow_from_dataframe(
            test,
            x_col='images',
            y_col='label',
            target_size=(128, 128),
            batch_size=512,
            class_mode='binary'
        )

        return [train_iterator, val_iterator]
