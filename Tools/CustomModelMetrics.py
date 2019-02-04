#!/usr/bin/env python

# METADATA

__author__ = "Adrian Rodriguez-Bazaga, Josep M. Porta"
__credits__ = ["Adrian Rodriguez-Bazaga", "Josep M. Porta"]
__version__ = "1.0.0"
__email__ = "adrianrodriguezbazaga@gmail.com"

from keras import backend as K

def m_recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def m_precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def m_f1(y_true, y_pred):
    precision = m_precision(y_true, y_pred)
    recall = m_recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))