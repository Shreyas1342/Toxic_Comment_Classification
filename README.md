# Toxic Comment Classification

This repository contains a machine learning project for classifying toxic comments. The goal of this project is to build a model that can accurately identify and classify toxic comments in online discussions.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The project uses the [Toxic Comment Classification Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) from Kaggle. This dataset consists of a large number of comments from various online platforms, labeled with different types of toxicity, such as toxic, severe toxic, obscene, threat, insult, and identity hate.

## Installation
1. Navigate to the project directory:
   ```
   cd toxic-comment-classification
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Preprocess the dataset:
   ```
   python preprocess.py
   ```
   This script performs text preprocessing tasks, such as removing special characters, tokenizing, and removing stop words.

2. Train the model:
   ```
   python train.py
   ```
   This script trains the toxic comment classification model using logistic regression and random forest algorithms on the preprocessed dataset.

3. Test the model:
   ```
   python test.py
   ```
   This script evaluates the trained model on a test set and displays the classification metrics.

## Model
The project uses logistic regression and random forest classifiers for toxic comment classification. These algorithms are commonly used for text classification tasks and have been shown to perform well in various scenarios. The models are trained using the preprocessed dataset and optimized to minimize the classification error.

## Evaluation
The model's performance is evaluated using common classification metrics, including accuracy, precision, recall, F1 score, and Area Under the ROC Curve (AUC). For logistic regression, the training AUC score is 0.99 and the testing AUC score is 0.97. These high AUC scores indicate that the model has a strong ability to discriminate between toxic and non-toxic comments.

## Contributing
Contributions to this project are welcome. If you have any ideas or suggestions, feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
