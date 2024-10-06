import warnings
import os
from visualisation import Visualisation
from training import Training
from model import Model
from classification import Classification

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

    main_path = ("C:\\Users\\User\\repos\\Dogs-vs-Cats-Image-Classification\\"
                 "Dogs-vs-Cats-Image-Classification")

    visual = Visualisation()
    visual.add_label_by_path(training_pet_images_path)

    df = visual.get_modified_dataframe()
    visual.validate_data()
    visual.display_images_of_dogs_or_cats('dogs')
    visual.display_images_of_dogs_or_cats('cats')
    visual.count_plot()

    training = Training(df)
    iterators = training.get_iterators()

    "Best results after 15 iterations of training -> model_training_2.h5 - Model"

    folder_file_path = '\\training_models_statistic\\'
    iteration_number = 2  # It started with 10 iterations, after that only +5 iterations
    end_file_path_loading = 'model_training_2.h5'
    end_file_path_saving = 'model_training.h5'

    model = Model(iterators[0], iterators[1])
    # model_training = model.load_model(main_path + folder_file_path + end_file_path_loading)
    history = model.create_model_or_continue_training(15)

    complete_file_path = main_path + folder_file_path + end_file_path_saving
    while os.path.exists(complete_file_path):
        complete_file_path = main_path + folder_file_path + 'model_training_' + str(iteration_number) + '.h5'
        iteration_number += 1

    model_training = model.save_model(complete_file_path)
    # Visualisation.show_training_history(history)

    classification = Classification()
    classification.import_own_pictures(main_path + '\\random_pet_images')
    classification.classify_own_pictures(model_training)
