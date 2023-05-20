# Logistic Regression for Credit Risk Prediction

This code implements logistic regression for predicting credit risk based on a German credit dataset. It uses one-hot encoding to preprocess the dataset and trains a logistic regression model using gradient descent.

## Dependencies
- Python 3.x
- pandas
- numpy
- requests
- zipfile
- sklearn

## Dataset
The code downloads a zip file from the Stanford University website and extracts the dataset file named `german_credit_data.csv`. This dataset contains information about credit applicants, including their attributes and credit risk labels.

## Preprocessing
The code performs the following preprocessing steps on the dataset:
- Dropping unneeded columns (`Unnamed: 0`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`, `Age`)
- Converting categorical variables to binary using one-hot encoding:
  - `Sex` is converted to `female` (1) and `male` (0)
  - `Housing` is converted to `own` (0), `rent` (1), and `free` (2)
  - `Purpose` is converted to multiple binary columns based on different purposes
  - `Risk` (target variable) is converted to `good` (1) and `bad` (0)

## Training and Evaluation
The code splits the preprocessed dataset into training, development, and test sets. It then defines several helper functions for logistic regression, including `sigmoid` (sigmoid activation function), `compute_logistic_regression` (computes logistic regression predictions), `get_loss` (calculates the loss function), `get_weight_gradient` (calculates the gradient of the weights), and `get_bias_gradient` (calculates the gradient of the bias).

The logistic regression model is trained using the training data, and the model is evaluated on the development data at regular intervals. The training loop performs gradient descent to update the weights and bias. The number of training epochs, learning rate, weights, and bias can be adjusted.

After training, the model is evaluated on the test data using the `evaluate_model` function, which calculates evaluation metrics such as accuracy, precision, and recall.

## Results
The code performs additional analysis on the test data. It selects only the examples with a female gender and computes the logistic regression predictions. It counts the number of females with approved loans. Then, it changes the gender of all female examples to male and computes the predictions again. It counts the number of females (with gender changed) with approved loans.

## Usage
To use this code, ensure that the required dependencies are installed. Run the code, which will download the dataset, preprocess it, train the logistic regression model, and evaluate its performance. The results will be printed, including evaluation metrics and loan approval statistics for females and females with changed gender.
