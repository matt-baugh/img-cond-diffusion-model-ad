"""
Author: Felix Meissen - https://github.com/FeliMe
Optimal DICE functions optimised by Matthew Baugh - https://github.com/matt-baugh

"""
from typing import Tuple
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def compute_average_precision(predictions, targets):
    """
    Compute Average Precision
    Args:
        predictions (torch.Tensor): Anomaly scores
        targets (torch.Tensor): Segmentation map or target label, must be binary
    """
    if (targets - targets.int()).sum() > 0.:
        raise RuntimeError("targets for AP must be binary")
    ap = average_precision_score(targets.reshape(-1), predictions.reshape(-1))
    return ap


def compute_auroc(predictions, targets) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve
    Args:
        predictions (torch.Tensor): Anomaly scores
        targets (torch.Tensor): Segmentation map or target label, must be binary
    """
    if (targets - targets.int()).sum() > 0.:
        raise RuntimeError("targets for AUROC must be binary")
    auc = roc_auc_score(targets.reshape(-1), predictions.reshape(-1))
    return auc


def compute_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the Sorensen-Dice coefficient:

    dice = 2 * TP / (2 * TP + FP + FN)

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Check if predictions and targets are binary
    if not np.all(np.logical_or(preds == 0, preds == 1)):
        raise ValueError('Predictions must be binary')
    if not np.all(np.logical_or(targets == 0, targets == 1)):
        raise ValueError('Targets must be binary')

    # Compute Dice
    dice = 2 * np.sum(preds[targets == 1]) / \
        (np.sum(preds) + np.sum(targets))

    return dice


def compute_dice_at_nfpr(preds: np.ndarray, targets: np.ndarray,
                         max_fpr: float = 0.05) -> float:
    """
    Computes the Sorensen-Dice score at 5% FPR.

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param max_fpr: Maximum false positive rate.
    :param n_threshs: Maximum number of thresholds to check.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Find threshold for 5% FPR
    fpr, _, thresholds = roc_curve(targets.reshape(-1), preds.reshape(-1))
    t = thresholds[max(0, fpr.searchsorted(max_fpr, 'right') - 1)]

    # Compute Dice
    return compute_dice(np.where(preds > t, 1, 0), targets)


def compute_thresh_at_nfpr(preds: np.ndarray, targets: np.ndarray,
                           max_fpr: float = 0.05) -> float:
    """
    Computes the threshold at 5% FPR.

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param max_fpr: Maximum false positive rate.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Find threshold for 5% FPR
    fpr, _, thresholds = roc_curve(targets.reshape(-1), preds.reshape(-1))
    t = thresholds[max(0, fpr.searchsorted(max_fpr, 'right') - 1)]

    # Return threshold
    return t


def compute_best_dice(preds: np.ndarray, targets: np.ndarray) \
        -> Tuple[float, float]:
    """
    Compute the best dice score for n_thresh thresholds.

    :param predictions: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    """
    return compute_average_precision_and_optimal_dice(preds, targets)[1:]

def compute_average_precision_and_optimal_dice(preds: np.ndarray, targets: np.ndarray) -> tuple[float, float]:

    preds, targets = np.array(preds).flatten(), np.array(targets).flatten()
    assert np.array_equal(np.unique(targets), [
                          0, 1]), f"Targets must be binary: {np.unique(targets)}"

    precision, recall, thresholds = precision_recall_curve(targets, preds)

    ap_score = -np.sum(np.diff(recall) * np.array(precision)[:-1])

    with np.errstate(divide='ignore', invalid='ignore'):
        dice_scores = 2 * precision * recall / (precision + recall)
    best_dice_i = np.nanargmax(dice_scores)
    return ap_score, dice_scores[best_dice_i], thresholds[best_dice_i]
