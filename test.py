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

def svm_train_primal(data_train, label_train, regularisation_para_C):
    N, num_features = data_train.shape
    w = cp.Variable(num_features)
    b = cp.Variable()
    xi = cp.Variable(N)

    objective = cp.Minimize(0.5 * cp.norm(w,2)**2 + (regularisation_para_C / N) * cp.sum(xi))
#     constraints = [label_train[i] * (w @ data_train[i] + b) >= 1 - xi[i] for i in range(N)]
#     constraints += [xi[i] >= 0 for i in range(N)]
    constraints = [cp.multiply(label_train, data_train @ w + b) >= 1 - xi, xi >= 0]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    svm_model = {'w': w.value, 'b': b.value}
    print("sum of w:", np.sum(w.value))
    print("b:", b.value)
    return svm_model

def svm_predict_primal(data_test, label_test, svm_model):
    w = svm_model['w']
    b = svm_model['b']
    predictions = np.sign(data_test @ w + b)
    accuracy = np.mean(predictions == label_test)
    return accuracy

regularisation_para_C = 100
svm_model = svm_train_primal(data_train, label_train, regularisation_para_C)
test_accuracy = svm_predict_primal(data_test, label_test, svm_model)

# w_sum = np.sum(svm_model['w'])

# print("Solution of b:", svm_model['b'])
# print("Sum of all dimensions of w solution:", w_sum)
# print("Test accuracy:", test_accuracy)