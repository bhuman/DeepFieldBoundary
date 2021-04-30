import numpy as np


def get_field_boundary_y(label, x):
    if label[1] == -1:
        return max(0, min(label[0] + x * (label[3] - label[0]), 1))
    elif x < label[1]:
        return max(0, min(label[0] + x * (label[2] - label[0]) / label[1], 1))
    else:
        return max(0, min(label[2] + (x - label[1]) * (label[3] - label[2]) / (1 - label[1]), 1))


def get_column_wise_y(label, width):
    return np.asarray([get_field_boundary_y(label, (0.5 + j) / width) for j in range(width)])


def draw_label(axes, image, label, color='r'):
    x = np.linspace(0, image.shape[1] - 1, num=image.shape[1])
    axes.plot(x, label * (image.shape[0] - 1), color)


def draw_prediction(axes, image, prediction):
    x = np.linspace(0, image.shape[1] - 1, num=prediction.shape[0])
    if len(prediction.shape) == 1:
        axes.plot(x, np.clip(prediction, 0, 1) * (image.shape[0] - 1), 'y')
    else:
        axes.plot(x, np.clip(prediction[..., 0], 0, 1) * (image.shape[0] - 1), 'y')
        # Draw uncertainty information as +-sigma corridor (1 / square is here because the predicted value is the square-root of the inverse of sigma).
        axes.plot(x, np.clip(prediction[..., 0] + 1 / np.square(prediction[..., 1]), 0, 1) * (image.shape[0] - 1), ':y')
        axes.plot(x, np.clip(prediction[..., 0] - 1 / np.square(prediction[..., 1]), 0, 1) * (image.shape[0] - 1), ':y')
