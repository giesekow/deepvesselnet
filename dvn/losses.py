from __future__ import print_function
from keras.datasets import mnist
from keras.utils import np_utils
from keras import layers as KL
from keras import backend as K
import numpy as np
import tensorflow as tf

def _categorical_crossentropy(target, output, from_logits=False, axis=-1):
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * tf.log(output), axis)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,logits=output)

def categorical_crossentropy(axis=-1):
    def loss(y_true, y_pred):
        return K.mean(_categorical_crossentropy(y_true, y_pred, axis=axis))
    return loss

def soft_dice(y_true, y_pred):
    smooth = 1
    intersection = K.sum(K.abs(y_true * y_pred), axis=1)
    coeff = (2. *  intersection + smooth) / (K.sum(K.square(y_true),1) + K.sum(K.square(y_pred),1) + smooth)

    return (1. - coeff)

def weighted_categorical_crossentropy(axis=1,from_logits=False,classes=2):
    def loss(y_true, y_pred):
        L = _categorical_crossentropy(target=y_true,output=y_pred,axis=axis,from_logits=from_logits)
        _epsilon = K.epsilon()
        y_true_p = K.argmax(y_true, axis=axis)
        for c in range(classes):
            c_true = K.cast(K.equal(y_true_p, c), K.dtype(y_pred))
            w = 1. / (K.sum(c_true))# + _epsilon)
            C = K.sum(L * c_true * w) if c == 0 else C + K.sum(L * c_true * w)

        return C

    return loss

def weighted_categorical_crossentropy_with_fpr(axis=1,from_logits=False,classes=2, threshold=0.5):
    def loss(y_true, y_pred):
        L = _categorical_crossentropy(target=y_true,output=y_pred,axis=axis,from_logits=from_logits)
        _epsilon = K.epsilon()
        y_true_p = K.argmax(y_true, axis=axis)
        y_pred_bin = K.cast(K.greater_equal(y_pred, threshold), K.dtype(y_true)) if from_logits else K.argmax(y_pred, axis=axis)
        y_pred_probs = y_preds if from_logits else K.max(y_pred, axis=axis)
        for c in range(classes):
            c_true = K.cast(K.equal(y_true_p, c), K.dtype(y_pred))
            w = 1. / (K.sum(c_true) + _epsilon)
            C = K.sum(L * c_true * w) if c == 0 else C + K.sum(L * c_true * w)

            # Calc. FP Rate Correction
            c_false_p = K.cast(K.not_equal(y_true_p, c), K.dtype(y_pred)) * K.cast(K.equal(y_pred_bin, c), K.dtype(y_pred)) # Calculate false predictions
            gamma = 0.5 + (K.sum(K.abs((c_false_p * y_pred_probs) - 0.5)) / (K.sum(c_false_p) + _epsilon)) # Calculate Gamme
            wc = w * gamma # gamma / |Y+|
            C = C + K.sum(L * c_false_p * wc) # Add FP Correction

        return C

    return loss
