from tensorflow.keras import backend as K
import numpy as np


def mean_absolute_error(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    if len(y_pred.shape) > 2:
        y_pred = y_pred[..., 0]
    return K.mean(K.abs(y_pred - y_true))

# ----------------------------------------------------------------------------------------------------------------------


def combined_loss(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)

    diff = y_pred[..., 0] - y_true
    omega = K.square(y_pred[..., 1])  # This square actually belongs in the model.

    return K.mean(_neg_log_gaussian_factor(omega) + _neg_gaussian_exp(diff, omega) + K.abs(diff))


def neg_log_likelihood(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)

    diff = y_pred[..., 0] - y_true
    omega = K.square(y_pred[..., 1])  # This square actually belongs in the model.

    return K.mean(_neg_log_gaussian_factor(omega) + _neg_gaussian_exp(diff, omega))


def chi_squared(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)

    diff = y_pred[:, :, 0] - y_true
    omega = K.square(y_pred[..., 1])  # This square actually belongs in the model.

    return 2 * K.mean(_neg_gaussian_exp(diff, omega))

# ----------------------------------------------------------------------------------------------------------------------


def _neg_log_gaussian_factor(omega):
    return K.cast_to_floatx(0.5 * np.log(2 * np.pi)) - K.log(omega + K.epsilon())


def _neg_gaussian_exp(diff, omega):
    return 0.5 * K.square(diff) * K.square(omega)



