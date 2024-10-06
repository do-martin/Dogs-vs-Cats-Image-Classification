import os

import numpy as np
from keras.preprocessing import image


class Classification:
    def __init__(self):
        self.testing_array = []
        self.full_paths = []
        self.testing_img = []
        self.prediction_array = []

    def import_own_pictures(self, random_pictures_path):
        for i_path in os.listdir(random_pictures_path):
            full_path = random_pictures_path + '\\' + i_path
            if os.path.isfile(full_path):
                self.full_paths.append(full_path)
                img = image.load_img(full_path, target_size=(128, 128))
                self.testing_img.append(img)

    def classify_own_pictures(self, model):
        iteration = 0
        true_calculation = 0
        false_calculation = 0
        false_calculation_path = []

        for i in range(len(self.testing_img) - 1):
            if os.path.isfile(self.full_paths[i]):
                iteration += 1
                img = image.load_img(self.full_paths[i], target_size=(128, 128))
                img_array = image.img_to_array(img) / 255.  # convert img to Numpy-Array
                img_array = np.expand_dims(img_array, axis=0)  # add dimension

                prediction = model.predict(img_array)  # predict
                self.prediction_array.append(prediction)

                filename = os.path.basename(self.full_paths[i])

                if self.prediction_array[i] < 0.5:
                    print("The picture is a cat.")
                    if ('katz' in filename or 'cat' in filename or 'Katz' in filename
                            or 'Cat' in filename or 'Katze' in filename):
                        true_calculation += 1
                    else:
                        false_calculation += 1
                        false_calculation_path.append(self.full_paths[i])

                else:
                    if ('hund' in filename or 'dog' in filename
                            or 'Hund' in filename or "Dog" in filename):
                        true_calculation += 1
                    else:
                        false_calculation += 1
                        false_calculation_path.append(self.full_paths[i])
                    print("The picture is a dog.")
                print(self.prediction_array[i])
                print(self.full_paths[i])
                print('----------------')

            input_path = []
            label = []
        print(iteration)
        print(true_calculation)
        print(false_calculation)
        print(false_calculation_path)
