from keras.datasets import mnist
from keras.utils import np_utils
from keras import layers as KL
from keras import backend as K
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_recall_curve

def dice_score(y_true, y_pred):
    return f1_score(y_true.flatten(), y_pred.flatten())

def dice_information(y_true, y_pred):
    prec, rec, thres = precision_recall_curve(y_true.flatten(), y_pred.flatten())
    f1 = (2. * prec * rec) / (prec + rec)
    ind = np.argmax(f1)
    return prec[ind], rec[ind], f1[ind], thres[ind]

def threshold_accuracy(threshold=0.5):
    def metric(y_true, y_pred):
        pred = K.cast(K.greater_equal(y_pred, threshold),'int32')
        return K.mean(K.equal(K.cast(y_true, 'int32'), pred))
    return metric

def categorical_accuracy(axis=-1):
    def accuracy(y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=axis),K.argmax(y_pred, axis=axis)))

    return accuracy

def dice(smooth=1):
    def metric(y_true, y_pred):
        intersection = K.sum(y_true * y_pred, axis=list(range(1, K.ndim(y_true))))
        union = K.sum(y_true, axis=list(range(1, K.ndim(y_true)))) + K.sum(y_pred, axis=list(range(1, K.ndim(y_true))))
        return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return metric
