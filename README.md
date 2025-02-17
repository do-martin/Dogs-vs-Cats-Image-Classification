# Dogs vs Cats Image Classification

The **"Dogs vs Cats Image Classification"** project uses a Convolutional Neural Network (CNN) to classify images of dogs and cats. This project demonstrates the application of deep learning for automated image processing and offers a user-friendly interface for instant classification.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- pip (Python package installer)

### Clone the Repository

1. Clone the repository:
   ```bash
   git clone https://github.com/do-martin/Dogs-vs-Cats-Image-Classification.git
   cd Dogs-vs-Cats-Image-Classification

### Install Required Packages

It is recommended to create a virtual environment first. Ensure that you have Python 3.11 or below installed:

#### Your default Python version is 3.11 or below: 

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

#### Alternative: Specify the path to Python 3.11:

```bash
C:\Path\to\python311\python.exe -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## Usage

The project is implemented in a Jupyter Notebook (.ipynb). Open the notebook in Jupyter Lab, Jupyter Notebook, or Visual Studio Code (VSCode) to explore and run the code.

## Model Evaluation

To facilitate an accurate assessment of the model's performance, please ensure that the folder named `images_test` is present. This folder should contain a diverse set of images featuring both dogs and cats. These images are exclusively intended for evaluating the model's accuracy and must not be utilized during the training phase.

### Guidelines for Sample Images

- The images should be in JPEG or PNG format.
- Ensure that each image name is unique to avoid any confusion.
- Use these images exclusively for evaluation purposes and not for training the model.

## Required Libraries

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn

You can find the specific library versions in the `requirements.txt` file.

## Disclaimer for Links

The links to external websites included in this project are provided for informational purposes only. I take no responsibility for the content or availability of these websites. The use of these links is at your own risk.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Special thanks to the developers of TensorFlow and Keras for their contributions to the field of deep learning.
- Inspiration from various Kaggle competitions that focus on image classification.
