from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
import numpy as np

def eval(y_true, y_pred, num_classes):
    y_true = label_binarize(y_true, classes=list(range(num_classes)))
    y_pred = label_binarize(y_pred, classes=list(range(num_classes)))

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true.ravel(), y_pred.ravel()
    )
    average_precision["micro"] = average_precision_score(y_true, y_pred, average="micro")
    return np.mean(precision["micro"]), np.mean(recall["micro"]), np.mean(average_precision["micro"])
