import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Utils"))

import cv2
import color_conversion
import h5py
import hdf5_utils
import image_augmentation
import image_loading
import label_utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import warnings
warnings.filterwarnings("ignore", message='Attempting to set identical left == right == 0 results in singular transformations;')
warnings.filterwarnings("ignore", message='Toggling axes navigation from the keyboard is deprecated since 3.3 and will be removed two minor releases later.')

from argparse import ArgumentParser
from matplotlib.widgets import Slider

plt.style.use('seaborn')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='The dataset which should be inspected')
    args = parser.parse_args()

    data = h5py.File(args.dataset, 'r')
    datasets = [dataset.rsplit('/', 1)[0] for dataset, key in hdf5_utils.h5py_dataset_iterator(data) if key == 'images']

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(10, 4, figure=fig)

    ax_img = fig.add_subplot(gs[0:8, :])
    ax_img.axis('off')
    ax_slider_dataset = fig.add_subplot(gs[8, :])
    ax_slider_image = fig.add_subplot(gs[9, :])

    slider_color = (71/255, 151/255, 149/255)
    slider_dataset = Slider(ax_slider_dataset, 'Dataset Index', 0, len(datasets) - 1, valinit=0, valfmt='%i', color=slider_color)
    slider_image = Slider(ax_slider_image, 'Image Index', 0, 1, valinit=0, valfmt='%i', color=slider_color)

    current_dataset = data[datasets[0]]
    im_ax_img = ax_img.imshow(image_loading.load_binary_image(current_dataset['images'][0]))
    augmentation, yuv_color_space = False, False

    def dataset_slider_changed(val):
        global current_dataset
        current_dataset = data[datasets[int(val)]]

        fig.suptitle(f'{datasets[int(val)]}', fontsize=10, ha='center')
        slider_image.valmax = current_dataset['images'].shape[0] - 1
        slider_image.ax.set_xlim(slider_image.valmin, slider_image.valmax)
        slider_image.set_val(0)

    def image_slider_changed(val):
        ax_img.lines = []
        image, label = image_loading.load_binary_image(current_dataset['images'][int(val)]), current_dataset['labels'][int(val)]
        if augmentation:
            image, label = image_augmentation.augmentation(image[np.newaxis, ...], label[np.newaxis, ...])
            image, label = np.squeeze(image), np.squeeze(label)
        if yuv_color_space:
            image = color_conversion.bgr2yuv(color_conversion.rgb2bgr(image))
        im_ax_img.set_data(image)
        label_utils.draw_label(ax_img, image, label_utils.get_column_wise_y(label, image.shape[1]))

        fig.canvas.draw()

    def arrow_key_image_control(event):
        global augmentation, yuv_color_space

        ind_image = int(slider_image.val)
        if event.key in ['left', 'right']:
            slider_image.set_val(max(ind_image-1, 0)) if event.key == 'left' else slider_image.set_val(min(ind_image+1, int(slider_image.valmax)))
        elif event.key == 'a':
            augmentation = not augmentation
            slider_image.set_val(slider_image.val)
        elif event.key == 'c':
            yuv_color_space = not yuv_color_space
            slider_image.set_val(slider_image.val)
        elif event.key == 'p':
            displayed_img = im_ax_img.get_array()
            for line in ax_img.lines:
                x, y = line.get_xdata(), line.get_ydata()
                cv2.polylines(displayed_img, [np.asarray(np.vstack([x, y]).T).astype(np.int32)], False, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.imwrite(f"{datasets[int(slider_dataset.val)].replace('/', '_')}_{ind_image}.png", cv2.cvtColor(displayed_img, cv2.COLOR_BGR2RGB))

    slider_dataset.on_changed(dataset_slider_changed)
    slider_image.on_changed(image_slider_changed)
    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)

    dataset_slider_changed(0)

    plt.show()
