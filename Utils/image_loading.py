import io
import numpy as np

from PIL import Image


def load_pgm(filepath):
    with open(filepath, 'rb') as pgm_file:
        return np.asarray([[int(value) for value in row.split()] for row in pgm_file.readlines()[3:]])


def load_binary_image(data):
    stream = io.BytesIO(data)
    img = np.asarray(Image.open(stream))
    stream.close()
    return img
