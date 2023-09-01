import cvxpy as cp
import numpy as np
import pandas as pd

def load_and_split_data(train_path="train.csv", test_path="test.csv"):
    """
    Load data from CSV files and split into training, validation, and test sets.
    
    Parameters:
    - train_path: Path to the training CSV file.
    - test_path: Path to the testing CSV file.
    
    Returns:
    - train_data, train_labels: Training data and labels.
    - val_data, val_labels: Validation data and labels.
    - test_data, test_labels: Test data and labels.
    """
    
    # Load train.csv and split
    train_df = pd.read_csv(train_path, header=None)

    train_data = train_df.iloc[:4000, 1:].values
    train_labels = train_df.iloc[:4000, 0].values

    # print("The first 10 labels in the training set are: ", train_labels[:10])
    # print("The first 10 features in the training set are: ", train_data[:10])    
    
    val_data = train_df.iloc[4000:, 1:].values
    val_labels = train_df.iloc[4000:, 0].values
    
    # Load test.csv
    test_df = pd.read_csv(test_path, header=None)
    test_data = test_df.iloc[:, 1:].values
    test_labels = test_df.iloc[:, 0].values

    test_labels = test_labels * 2 - 1
    val_labels = val_labels * 2 - 1
    train_labels = train_labels * 2 - 1
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

# Load data
train_data, train_labels, val_data, val_labels, test_data, test_labels = load_and_split_data()

def predict(data, w, b):
    """
    Predict labels for given data using weight vector w and bias b.
    
    Parameters:
    - data: Data matrix.
    - w: Weight vector.
    - b: Bias scalar.
    
    Returns:
    - labels: Predicted labels.
    """
    return np.sign(data @ w + b)

# %%
def svm_train_primal(data_train, label_train, regularisation_para_C):
    """
    Train a linear SVM in the primal form using cvxpy.
    
    Parameters:
    - data_train: Training data matrix of shape (N, d)
    - label_train: Training labels vector of shape (N, )
    - regularisation_para_C: Regularization parameter
    
    Returns:
    - w: Weight vector of shape (d, )
    - b: Bias (scalar)
    """
    
    N, d = data_train.shape
    w = cp.Variable(d)
    b = cp.Variable()
    xi = cp.Variable(N)
    
    # Objective function
    objective = cp.Minimize(0.5 * cp.norm(w,2)**2 + (regularisation_para_C/N) * cp.sum(xi))
    
    # Constraints
    constraints = [cp.multiply(label_train, data_train @ w + b) >= 1 - xi, xi >= 0]
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return w.value, b.value


# %%
def is_positive_semidefinite(K):
    """
    Check if matrix K is positive semidefinite.
    
    Parameters:
    - K: Square matrix
    
    Returns:
    - True if K is positive semidefinite, False otherwise.
    """
    eigenvalues = np.linalg.eigvalsh(K)
    return np.all(eigenvalues >= 0)

# Construct the kernel (gram matrix) for the training data
K = np.outer(train_labels, train_labels) * (train_data @ train_data.T)

# Check if K is positive semidefinite
if is_positive_semidefinite(K):
    print("Matrix K is positive semidefinite.")
else:
    print("Matrix K is NOT positive semidefinite. Minimum eigenvalue:", np.min(np.linalg.eigvalsh(K)))


# %%
def svm_train_dual(data_train, label_train, regularisation_para_C):
    regularization_term = 1e-5
    N, _ = data_train.shape
    alpha = cp.Variable(N)
    
    # Construct the kernel (gram matrix) with regularization
    K = np.outer(label_train, label_train) * (data_train @ data_train.T)
    K += np.eye(N) * regularization_term
    
    # Explicitly compute the quadratic term for the objective using quad_form
    quadratic_term = 0.5 * cp.quad_form(alpha, K)
    
    # Objective
    objective = cp.Minimize(-1 * cp.sum(alpha) + quadratic_term)
    
    # Constraints
    constraints = [
        cp.sum(cp.multiply(alpha, label_train)) == 0,
        alpha >= 0,
        alpha <= regularisation_para_C / N
    ]
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return alpha.value


# %%
def svm_predict_primal(data, labels, svm_model):
    """
    Predict using the primal SVM model and return accuracy.

    Parameters:
    - data: Data matrix (N x D)
    - labels: Ground truth labels (N,)
    - svm_model: Trained SVM model with 'w' and 'b' as keys

    Returns:
    - Accuracy of predictions
    """
    # Extract w and b from the model
    w = svm_model['w']
    b = svm_model['b']

    # Predict
    preds = np.sign(np.dot(data, w) + b)
    
    # Calculate accuracy
    accuracy = np.mean(preds == labels)
    return accuracy


# %%
def primal_svm_driver():
    # Load data
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_and_split_data()
    
    # Train SVM
    w, b = svm_train_primal(train_data, train_labels, 100)

    # Store w and b in svm_model
    svm_model = {'w': w, 'b': b}
    
    # Predict on validation and test sets using svm_predict_primal
    val_accuracy = svm_predict_primal(val_data, val_labels, svm_model)
    test_accuracy = svm_predict_primal(test_data, test_labels, svm_model)
    
    # Print results
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    return svm_model


# %%
def dual_svm_driver():
        # Load the data (assuming load_and_split_data is defined)
    train_data, train_labels, _, _, _, _ = load_and_split_data()

    # # Convert labels from {0, 1} to {-1, 1}
    # train_labels = 2 * train_labels - 1

    # Train SVM in dual form
    alpha_values = svm_train_dual(train_data, train_labels, 100)

    # Compute the weight vector w
    w = np.sum((alpha_values * train_labels)[:, None] * train_data, axis=0)

    # Compute the bias b using a support vector (any example where 0 < alpha < C/N can be a support vector)
    support_vector_indices = np.where((alpha_values > 1e-5) & (alpha_values < (100/train_data.shape[0])))[0]
    if len(support_vector_indices) > 0:
        sv_index = support_vector_indices[0]
        b = train_labels[sv_index] - np.dot(w, train_data[sv_index])
    else:
        b = 0

    # Store w and b in svm_model
    svm_model = {'w': w, 'b': b}

    # Predict on validation and test sets using svm_predict_primal
    val_accuracy = svm_predict_primal(val_data, val_labels, svm_model)
    test_accuracy = svm_predict_primal(test_data, test_labels, svm_model)
    
    # Print results
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    return svm_model


# %%
def compute_primal_from_dual(alpha, data_train, label_train, regularisation_para_C):
    """
    Compute the primal SVM solution (w*, b*) from the dual solution alpha*.

    Parameters:
    - alpha: Dual solution (N,)
    - data_train: Training data matrix (N x D)
    - label_train: Training labels (N,)
    - regularisation_para_C: Regularization parameter C

    Returns:
    - w: Weight vector (D,)
    - b: Bias scalar
    """
    # Compute w* from alpha*
    w = np.sum((alpha * label_train)[:, None] * data_train, axis=0)

    # Compute b* using a support vector
    # Find a support vector index (any example where 0 < alpha < C/N)
    support_vector_indices = np.where((alpha > 1e-5) & (alpha < (regularisation_para_C / data_train.shape[0])))[0]
    if len(support_vector_indices) > 0:
        sv_index = support_vector_indices[0]
        b = label_train[sv_index] - np.dot(w, data_train[sv_index])
    else:
        b = 0

    return w, b

# %%
# Run the main function
if __name__ == "__main__":
    alpha = svm_train_dual(train_data, train_labels, 100)
    w, b = compute_primal_from_dual(alpha, train_data, train_labels, 100)
    print("Primal from Dual:\tsum of w: ", np.sum(w))
    print("Primal from Dual:\tb: ", b)
    #calculate w abd b from primal form and compare with dual form
    svm_model = primal_svm_driver()
    


