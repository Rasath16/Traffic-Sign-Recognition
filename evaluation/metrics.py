"""Evaluation metrics for classification."""
from sklearn.metrics import accuracy_score, classification_report


def simple_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def report(y_true, y_pred, target_names=None):
    return classification_report(y_true, y_pred, target_names=target_names)
