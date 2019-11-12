"""
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import print_function

import keras
import tensorflow as tf
import keras.backend as K
from keras.losses import kullback_leibler_divergence


def categorical_loss(x1, x2):
    return -tf.reduce_sum(x1 * tf.log(x2), axis=len(x2.get_shape()) - 1)


def absolute_error_loss(x1, x2):
    return K.abs(x1 - x2)


def calculate_delta_errors(y_true, attention_weights, auxiliary_outputs, all_but_one_auxiliary_outputs,
                           loss_function):
    error_with_all_features = loss_function(y_true, auxiliary_outputs)

    delta_errors = []
    for all_but_one_auxiliary_output in all_but_one_auxiliary_outputs:
        error_without_one_feature = loss_function(y_true, all_but_one_auxiliary_output)
        # The error without the feature is an indicator as to how potent the left-out feature is as a predictor.
        delta_error = tf.maximum(error_without_one_feature - error_with_all_features, K.epsilon())
        delta_errors.append(delta_error)
    delta_errors = tf.stack(delta_errors, axis=-1)

    shape = K.int_shape(delta_errors)
    if len(shape) > 2:
        delta_errors = K.squeeze(delta_errors, axis=-2)
    delta_errors /= (K.sum(delta_errors, axis=-1, keepdims=True))

    # Ensure correct format.
    delta_errors = tf.clip_by_value(delta_errors, K.epsilon(), 1.0)
    attention_weights = tf.clip_by_value(attention_weights, K.epsilon(), 1.0)

    if len(attention_weights.shape) == 3:
        attention_weights = tf.squeeze(attention_weights, axis=-1)

    # NOTE: Without stop_gradient back-propagation would attempt to optimise the error_variance
    # instead of/in addition to the distance between attention weights and Granger causality index,
    # which is not desired.
    delta_errors = tf.stop_gradient(delta_errors)
    return delta_errors, attention_weights


def granger_causal_loss(y_true, y_pred, attention_weights, auxiliary_outputs, all_but_one_auxiliary_outputs,
                        loss_function):
    delta_errors, attention_weights = calculate_delta_errors(y_true,
                                                             attention_weights,
                                                             auxiliary_outputs,
                                                             all_but_one_auxiliary_outputs,
                                                             loss_function)
    return K.mean(kullback_leibler_divergence(delta_errors, attention_weights))


def repeat_output_loss(y_true, y_pred, outputs, main_loss):
    main_loss_fn = keras.losses.get(main_loss)
    all_outputs = tf.stack(outputs)
    y_true = tf.ones([len(outputs), 1, 1]) * y_true
    return K.sum(main_loss_fn(y_true, all_outputs), axis=0)
