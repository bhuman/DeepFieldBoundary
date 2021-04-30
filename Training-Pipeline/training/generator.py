import color_conversion
import h5py
import image_augmentation
import image_loading
import label_utils
import numpy as np

from PIL import Image
from tensorflow import keras


class DiscreteFieldBoundaryGenerator(keras.utils.Sequence):
    def __init__(self, dataset, groups, width, height, batch_size, shuffle, grayscale, augmentation, distributed_augmentation, ooid=False):
        self.batch_size, self.shuffle, self.grayscale, self.augmentation, self.distributed_augmentation = batch_size, shuffle, grayscale, augmentation, distributed_augmentation
        self.width, self.height = width, height
        self.ooid = ooid

        self.dataset = h5py.File(dataset, 'r')
        self.samples = []
        for group in groups:
            self.samples += [{'dataset': f'{group}/images', 'index': i, 'label': l} for i, l in enumerate(self.dataset[f'{group}/labels'])]
        self.samples = np.array(self.samples)
        self.indices = np.arange(len(self.samples), dtype=int)

        self.on_epoch_end()

    def _convert_image(self, image):
        image = Image.fromarray(image).resize((self.width, self.height), Image.NEAREST)
        if self.grayscale:
            image = image.convert('L')
            return np.array(image)[..., np.newaxis]
        return color_conversion.bgr2yuv(color_conversion.rgb2bgr(np.asarray(image)))

    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index, rooid=False):
        samples = self.samples[self.indices[index * self.batch_size:min(len(self.samples), (index + 1) * self.batch_size)]]

        images = np.asarray([image_loading.load_binary_image(self.dataset[sample['dataset']][sample['index']]) for sample in samples])
        annotations = [sample['label'] for sample in samples]
        if self.augmentation:
            images, annotations = image_augmentation.augmentation(images, annotations, self.distributed_augmentation)

        x = np.asarray([self._convert_image(images[i]) for i in range(images.shape[0])])
        y = np.array([[label_utils.get_field_boundary_y(label, (0.5 + j) / self.width) for j in range(self.width)] for label in annotations])

        if self.ooid or rooid:
            return x, y, images, np.asarray(annotations)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
