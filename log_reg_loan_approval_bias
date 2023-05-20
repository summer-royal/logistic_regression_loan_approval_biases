import pandas as pd
import numpy as np

np.random.seed(42)
np.set_printoptions(suppress=True)

import requests
import zipfile
import io

r = requests.get("http://web.stanford.edu/class/cs21si/resources/unit2_resources.zip")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
credit_df = pd.read_csv('unit2_resources/german_credit_data.csv')
# Remove unneeded columns from data.
credit_df = credit_df.drop('Unnamed: 0', 1).drop('Saving accounts', 1).drop('Checking account', 1).drop('Credit amount', 1).drop('Duration', 1).drop('Age', 1)

def one_hot_encode_dataset(credit_df):
    credit_df = credit_df.replace({'Sex': {'male': 0, 'female': 1}})
    credit_df = credit_df.replace({'Housing': {'own': 0, 'rent': 1, 'free': 2}})
    credit_df = credit_df.replace({'Purpose': {'car': 0, 
               'furniture/equipment': 1, 'radio/TV': 2, 
               'domestic appliances': 3, 'repairs': 4, 'education': 5, 
               'business': 6, 'vacation/others': 7}})
    credit_df = credit_df.replace({'Risk': {'good': 1, 'bad': 0}})
    enc = OneHotEncoder(categories='auto')
    enc.fit(credit_df.values)
    dataset = enc.transform(credit_df.values).toarray()
    
    # List of binary columns in the final data, where the last column is the
    # risk to be predicted.
    columns = ['female', 'job:1', 'job:2', 'job:3', 'job:4', 'job:5', 'job:6', 
               'housing:own', 'housing:rent', 'housing:free', 
               'purpose:car', 'purpose:furniture/equipment', 'purpose:radio/TV', 
               'purpose:domestic appliances', 'purpose:repairs', 
               'purpose:education', 'purpose:business', 
               'purpose:vacation/others', 'risk:good']
    
    # Convert back to dataframe for easy viewing.
    processed_credit_df = pd.DataFrame(dataset, columns=columns)
    
    X, y = dataset[:, :-1], dataset[:, -1]
    
    return shuffle(X, y, random_state=0), processed_credit_df

(X, y), processed_credit_df = one_hot_encode_dataset(credit_df)

X_train, X_dev, X_test = X[:800], X[800:900], X[900:]
y_train, y_dev, y_test = y[:800], y[800:900], y[900:]

print("Training data shape", X_train.shape, y_train.shape)
print("Dev data shape", X_dev.shape, y_dev.shape)
print("Test data shape", X_test.shape, y_test.shape)

def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))
  
def compute_logistic_regression(x, weights, bias):
  return sigmoid(np.dot(weights, x) + bias)
  
def get_loss(y, y_hat):
  return -1 * y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)
  
def get_weight_gradient(y, y_hat, x):
  return (y_hat - y) * x
  
def get_bias_gradient(y, y_hat, x):
  return y_hat - y
def evaluate_model(eval_data, weights, bias):
    num_examples = len(eval_data)
    total_correct = 0.0
    true_positives = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    for i in range(num_examples):
        x, y = eval_data[i]
        pred = compute_logistic_regression(x, weights, bias)
        
        total_correct += 1 if (pred > .5 and y == 1 or pred <= .5 and y == 0) else 0
        true_positives += 1 if pred > .5 and y == 1 else 0
        false_positives += 1 if pred > .5 and y == 0 else 0
        false_negatives += 1 if pred <= .5 and y == 1 else 0
    print("Evaluation accuracy: ", total_correct / num_examples)
    print("Precision: ", true_positives / (true_positives + false_positives))
    print("Recall: ", true_positives / (true_positives + false_negatives))
    print()
def fit_logistic_regression(training_data, dev_data, NUM_EPOCHS=50, LEARNING_RATE=0.0005):
    np.random.seed(42)
    weights = np.random.randn(18)
    bias = 0
    
    for epoch in range(NUM_EPOCHS):
        loss = 0
        for example in training_data:
            x, y = example
            y_hat = compute_logistic_regression(x, weights, bias)
            loss += get_loss(y, y_hat)

            dw = get_weight_gradient(y, y_hat, x)
            db = get_bias_gradient(y, y_hat, x)

            weights -= LEARNING_RATE * dw
            bias -= LEARNING_RATE * db
        if epoch % 10 == 0:
            print("Epoch %d, loss = %f" % (epoch, loss))   
            print("Evaluating model on dev data...")
            evaluate_model(training_data, weights, bias)
    return weights, bias
training_data = list(zip(X_train, y_train))
dev_data = list(zip(X_dev, y_dev))
weights, bias = fit_logistic_regression(training_data, dev_data)
test_data = list(zip(X_test, y_test))
evaluate_model(test_data, weights, bias)

X_test_female = X_test[X_test[:, 0] == 1]
print("There are %i females in the test set." % len(X_test_female))

results = [compute_logistic_regression(x, weights, bias) for x in X_test_female]
total_good = sum(1 if result > .5 else 0 for result in results)
print("%i females had loans approved." % total_good)


X_test_all_male = X_test_female.copy()
X_test_all_male[:, 0] = 0

results = [compute_logistic_regression(x, weights, bias) for x in X_test_all_male]
total_good = sum(1 if result > .5 else 0 for result in results)
print("%i females with gender changed had loans approved." % total_good)
