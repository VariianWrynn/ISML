import numpy as np
import pandas as pd
import cvxpy as cp

# Load training and testing data from CSV files
train_data = pd.read_csv('train.csv', header=None).values
test_data = pd.read_csv('test.csv', header=None).values

# Extract labels and features from the data
label_train = train_data[:4000, 0]
data_train = train_data[:4000, 1:]

label_val = train_data[4000:, 0]
data_val = train_data[4000:, 1:]

label_test = test_data[:, 0]
data_test = test_data[:, 1:]

def select_best_C(train_data, train_labels, val_data, val_labels):
    """
    Choose the best value of C using the validation set.

    Parameters:
    - train_data: Training data matrix
    - train_labels: Training labels
    - val_data: Validation data matrix
    - val_labels: Validation labels

    Returns:
    - best_C: Optimal value of C
    - best_accuracy: Accuracy on the validation set using the best C
    """
    C_values = [2**i for i in range(-10, 11)]
    best_accuracy = -0.1
    best_C = None
    
    for C in C_values:
        # Train SVM using the current value of C
        svm_model = svm_train_primal(train_data, train_labels, C)
        val_preds = svm_predict_primal(val_data, val_labels, svm_model)
        
        # Compute accuracy on the validation set
        accuracy = np.mean(val_preds == val_labels)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C
            
    return best_C, best_accuracy

# Using the above function
best_C, best_val_accuracy = select_best_C(train_data, train_labels, val_data, val_labels)
print(f"Best C: {best_C}, Validation Accuracy: {best_val_accuracy * 100:.2f}%")

# Train the SVM using the best C and report test accuracy
svm_model = svm_train_primal(train_data, train_labels, best_C)

test_preds = svm_predict_primal(test_data, test_labels, svm_model)
test_accuracy = np.mean(test_preds == test_labels)
print(f"Test Accuracy using best C: {test_accuracy * 100:.2f}%")