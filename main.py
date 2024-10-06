import os
import random
import warnings
import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from keras import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array

warnings.filterwarnings('ignore')

# -*- coding: utf-8 -*-
"""Dogs-vs-Cats-Image-Classification.py

##Dataset Information
The training archive contains 25,000 images of dogs and cats. Train your algorithm on these files and predict the labels
 for test1.zip
(1 = dog, 0 = cat).

""Download Dataset: "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-
8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    training_pet_images_path = ("C:\\Users\\User\\repos\\Dogs-vs-Cats-Image-Classification\\"
                                "Dogs-vs-Cats-Image-Classification\\pet_images")

    saving_path = ("C:\\Users\\User\\repos\\Dogs-vs-Cats-Image-Classification\\"
                   "Dogs-vs-Cats-Image-Classification")

    main_path = ("C:\\Users\\User\\repos\\Dogs-vs-Cats-Image-Classification\\"
                 "Dogs-vs-Cats-Image-Classification")

    """##Import Modules"""

    """## Create Dataframe for Input and Output"""

    input_path = []
    label = []

    for class_name in os.listdir(training_pet_images_path):
        for path in os.listdir(training_pet_images_path + '\\' + class_name):
            if class_name == 'cats':
                label.append(0)
            else:
                label.append(1)
            input_path.append(os.path.join(training_pet_images_path, class_name, path))
    print(input_path[0], label[0])

    df = pd.DataFrame()
    df['images'] = input_path
    df['label'] = label
    df = df.sample(frac=1).reset_index(drop=True)
    df.head()

    for i in df['images']:
        if '.jpg' not in i:
            print(i)

    l = []
    for image in df['images']:
        try:
            img = PIL.Image.open(image)
        except:
            l.append(image)
    print(l)

    """## Exploratory Data Analysis"""

    # to display grid of images 'Dogs'
    plt.figure(figsize=(25, 25))
    temp = df[df['label'] == 1]['images']
    start = random.randint(0, len(temp))
    files = temp[start:start + 25]

    for index, file in enumerate(files):
        plt.subplot(5, 5, index + 1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.title('Dogs')
        plt.axis('off')

    # to display grid of images 'Cats'
    plt.figure(figsize=(25, 25))
    temp = df[df['label'] == 0]['images']
    start = random.randint(0, len(temp))
    files = temp[start:start + 25]

    for index, file in enumerate(files):
        plt.subplot(5, 5, index + 1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.title('Cats')
        plt.axis('off')

    sns.countplot(df['label'])

    """## Create DataGenerator for the Images"""

    df['label'] = df['label'].astype('str')

    df.head()

    # input split

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train_generator = ImageDataGenerator(
        rescale=1. / 255,
        # normalization of images
        rotation_range=40,
        # augmention of images to avoid overfitting
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
        class_mode='binary')

    val_iterator = val_generator.flow_from_dataframe(
        test,
        x_col='images',
        y_col='label',
        target_size=(128, 128),
        batch_size=512,
        class_mode='binary')

    # # Laden des Modells
    # from keras.models import load_model
    # model = load_model('/content/mein_modell.h5')

    # # Weiteres Training des Modells
    # history = model.fit(train_iterator, epochs=1, validation_data=val_iterator)

    """## Model Creation"""

    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPool2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    #history = model.fit(train_iterator, epochs=10, validation_data=val_iterator)  # normally 10 epochs

    """## Save Model"""

    # model.save(random_pictures_path + '\\dog_vs_cat_calculation_model.h5')

    # Calling `save('my_model.keras')` creates a zip archive `my_model.keras`.
    # model.save(random_pictures_path + "\\dog_vs_cat_calculation_model.keras")

    # It can be used to reconstruct the model identically.
    # reconstructed_model = load_model("my_model.keras")

    """## Load Model"""

    model = load_model(main_path + '\\dog_vs_cat_calculation_model.h5')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # # Weiteres Training des Modells
    # history = model.fit(train_iterator, epochs=1, validation_data=val_iterator)

    history = model

    """## Visualization of Results"""

    """acc = history.history['accuracy']
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
    plt.show()"""

    """## Classificate own pictures

    """

    testing_array = []

    full_paths = []
    testing_img = []
    prediction_array = []

    # import images
    for i_path in os.listdir(main_path + '\\random_pet_images'):
        full_path = main_path + '\\random_pet_images\\' + i_path
        if os.path.isfile(full_path):
            full_paths.append(full_path)
            img = load_img(full_path, target_size=(128, 128))
            testing_img.append(img)

    # Laden Sie das Bild
    iteration = 0
    trueCalculation = 0
    falseCalculation = 0
    false_calculation_path = []

    for i in range(len(testing_img) - 1):
        if os.path.isfile(full_paths[i]):
            iteration = iteration + 1
            img = load_img(full_paths[i], target_size=(128, 128))
            img_array = img_to_array(img) / 255.  # convert img to Numpy-Array
            img_array = np.expand_dims(img_array, axis=0)  # add dimension

            prediction = model.predict(img_array)  # predict
            prediction_array.append(prediction)

            filename = os.path.basename(full_paths[i])

            if prediction_array[i] < 0.5:
                print("Das Bild ist wahrscheinlich eine Katze")
                if ('katz' in filename or 'cat' in filename
                        or 'Katz' in filename or 'Cat' in filename):
                    trueCalculation += 1
                else:
                    falseCalculation += 1
                    false_calculation_path.append(full_paths[i])

            else:
                if ('hund' in filename or 'dog' in filename
                        or 'Hund' in filename or 'Dog' in filename):
                    trueCalculation += 1
                else:
                    falseCalculation += 1
                    false_calculation_path.append(full_paths[i])
                print("Das Bild ist wahrscheinlich ein Hund")
            print(prediction_array[i])
            print(full_paths[i])
            print('----------------')

        input_path = []
        label = []

    print('Anzahl der Durchläufe: ' + str(iteration))
    print('Anzahl der richtigen Aussagen: ' + str(trueCalculation))
    print('Anzahl der falschen Aussagen: ' + str(falseCalculation))
    print('Pfad der falschen Aussage lautet: ' + str(false_calculation_path))
