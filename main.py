import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd


def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays
    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row.
    """
    data = pd.read_csv(filename)
    Y = data.iloc[:, -1].values
    X = data.iloc[:, :-1].values
    return X, Y


def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))


def plot_results(X, y_true, y_pred, title, accuracy, errors=None):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='bwr')
    plt.title('True Labels (' + title + ')')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='bwr')
    plt.title('Predicted Labels (' + title + ')')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.text(0.05, 0.95, 'Accuracy: %.4f' % accuracy, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    if errors is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(errors) + 1), errors, marker='o', color='b')
        plt.title('Adaboost Error During Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()

    plt.tight_layout()
    plt.show()


def main():
    """
    This code is called directly in PyCharm without command-line arguments.
    Modify the file paths here.
    """
    # Load the emails.csv file using pandas
    data = pd.read_csv('emails.csv')

    # Separate features (X) and labels (Y)
    X = data.iloc[:, 1:-1].values  # Assuming the first column is the email name, and the last column is the label
    Y = data.iloc[:, -1].values

    # Ensure the data is balanced during the split (80% train and 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # SVM
    svm_model = SVC()
    svm_model.fit(X_train, Y_train)

    Yhat_train_svm = svm_model.predict(X_train)
    Yhat_test_svm = svm_model.predict(X_test)
    # Adaboost
    num_trees = 10
    trees = []
    trees_weights = []
    N = len(Y_train)
    curr_weights = [1 / float(len(Y_train)) for _ in range(N)]
    errors = []

    for _ in range(num_trees):
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X_train, Y_train, sample_weight=curr_weights)
        y_pred = tree.predict(X_train)
        to_change = [i for i in range(N) if y_pred[i] != Y_train[i]]
        error = np.sum([curr_weights[i] for i in to_change])
        alpha = np.log((1 - error) / error) if error > 0 else 1
        curr_weights = [curr_weights[j] * np.exp(alpha) if j in to_change else curr_weights[j] for j in range(N)]
        curr_weights /= np.sum(curr_weights)  # Normalize the weights
        trees.append(tree)
        trees_weights.append(alpha)
        errors.append(error)

    Yhat_train_adaboost = np.sign(sum(alpha * tree.predict(X_train) for alpha, tree in zip(trees_weights, trees)))
    Yhat_test_adaboost = np.sign(sum(alpha * tree.predict(X_test) for alpha, tree in zip(trees_weights, trees)))

    acc_train_svm = accuracy_score(Y_train, Yhat_train_svm)
    acc_test_svm = accuracy_score(Y_test, Yhat_test_svm)
    acc_train_adaboost = accuracy_score(Y_train, Yhat_train_adaboost)
    acc_test_adaboost = accuracy_score(Y_test, Yhat_test_adaboost)

    # print("SVM Train Accuracy: %.4f" % acc_train_svm)
    # print("SVM Test Accuracy: %.4f" % acc_test_svm)
    # print("Adaboost Train Accuracy: %.4f" % acc_train_adaboost)
    # print("Adaboost Test Accuracy: %.4f" % acc_test_adaboost)

    # Calculate the difference in accuracy between SVM and Adaboost
    acc_diff_train = acc_train_svm - acc_train_adaboost
    acc_diff_test = acc_test_svm - acc_test_adaboost

    # Calculate the total error
    total_error = np.cumsum(errors)

    # Calculate additional evaluation metrics
    report_train_svm = classification_report(Y_train, Yhat_train_svm, target_names=['Not Spam', 'Spam'],
                                             output_dict=True)
    report_test_svm = classification_report(Y_test, Yhat_test_svm, target_names=['Not Spam', 'Spam'], output_dict=True)
    report_train_adaboost = classification_report(Y_train, Yhat_train_adaboost, target_names=['Not Spam', 'Spam'],
                                                  output_dict=True)
    report_test_adaboost = classification_report(Y_test, Yhat_test_adaboost, target_names=['Not Spam', 'Spam'],
                                                 output_dict=True)

    # Extract metrics from classification report
    metrics = ['recall', 'precision', 'f1-score']
    metrics_train_svm = [report_train_svm['weighted avg'].get(metric, 0) for metric in metrics]
    metrics_test_svm = [report_test_svm['weighted avg'].get(metric, 0) for metric in metrics]
    metrics_train_adaboost = [report_train_adaboost['weighted avg'].get(metric, 0) for metric in metrics]
    metrics_test_adaboost = [report_test_adaboost['weighted avg'].get(metric, 0) for metric in metrics]

    # Plotting the results
    labels = metrics
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width / 2, metrics_train_adaboost, width, label='Adaboost (Train)', color='green', alpha=0.5)
    ax.bar(x + width / 2, metrics_test_adaboost, width, label='Adaboost (Test)', color='blue', alpha=0.5)
    ax.bar(x + 3 * width / 2, metrics_train_svm, width, label='SVM (Train)', color='yellow', alpha=0.5)
    ax.bar(x + 5 * width / 2, metrics_test_svm, width, label='SVM (Test)', color='orange', alpha=0.5)

    ax.set_title('Metrics Comparison')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Calculate confusion matrices
    cm_train_svm = confusion_matrix(Y_train, Yhat_train_svm)
    cm_test_svm = confusion_matrix(Y_test, Yhat_test_svm)
    cm_train_adaboost = confusion_matrix(Y_train, Yhat_train_adaboost)
    cm_test_adaboost = confusion_matrix(Y_test, Yhat_test_adaboost)

    # Plotting the confusion matrices
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cm_train_svm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('SVM Confusion Matrix (Train)')
    plt.colorbar()
    tick_marks = np.arange(len(set(Y_train)))
    plt.xticks(tick_marks, ['Not Spam', 'Spam'], rotation=45)
    plt.yticks(tick_marks, ['Not Spam', 'Spam'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.subplot(1, 2, 2)
    plt.imshow(cm_test_svm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('SVM Confusion Matrix (Test)')
    plt.colorbar()
    tick_marks = np.arange(len(set(Y_test)))
    plt.xticks(tick_marks, ['Not Spam', 'Spam'], rotation=45)
    plt.yticks(tick_marks, ['Not Spam', 'Spam'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cm_train_adaboost, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Adaboost Confusion Matrix (Train)')
    plt.colorbar()
    tick_marks = np.arange(len(set(Y_train)))
    plt.xticks(tick_marks, ['Not Spam', 'Spam'], rotation=45)
    plt.yticks(tick_marks, ['Not Spam', 'Spam'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.subplot(1, 2, 2)
    plt.imshow(cm_test_adaboost, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Adaboost Confusion Matrix (Test)')
    plt.colorbar()
    tick_marks = np.arange(len(set(Y_test)))
    plt.xticks(tick_marks, ['Not Spam', 'Spam'], rotation=45)
    plt.yticks(tick_marks, ['Not Spam', 'Spam'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
