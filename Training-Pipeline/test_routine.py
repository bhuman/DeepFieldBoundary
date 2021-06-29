import sys
sys.path.append("../Utils")
sys.path.insert(0, './training')

import generator
import h5py
import hdf5_utils
import fitting
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
from tensorflow.keras.models import load_model

plt.style.use('seaborn')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='The dataset which should be inspected')
    parser.add_argument('model', type=str, help='The model used for predictions')
    args = parser.parse_args()

    # Load data information
    data = h5py.File(args.dataset, 'r')
    datasets = [dataset.rsplit('/', 1)[0] for dataset, key in hdf5_utils.h5py_dataset_iterator(data) if key == 'images']
    current_dataset, img_generator = -1, None

    # Load model
    model = load_model(args.model, compile=False)

    # Initialize the figure
    fig = plt.figure(figsize=(12, 8))
    gs0 = gridspec.GridSpec(10, 1, figure=fig, wspace=2)

    # Create visualization grid
    grid_dim = 3
    gs00 = gridspec.GridSpecFromSubplotSpec(grid_dim, grid_dim, subplot_spec=gs0[0:8])
    grid = [fig.add_subplot(gs00[i, j]) for i in range(grid_dim) for j in range(grid_dim)]
    for axis in grid:  axis.axis('off')
    dummy_image = np.zeros(image_loading.load_binary_image(data[datasets[0]]['images'][0]).shape)
    grid_axes = [ax.imshow(im) for ax, im in zip(grid, [dummy_image] * grid_dim * grid_dim)]

    # Sliders
    gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[8:])
    ax_slider_dataset = fig.add_subplot(gs01[0, :])
    ax_slider_batch = fig.add_subplot(gs01[1, :])
    slider_color = (71 / 255, 151 / 255, 149 / 255)
    slider_dataset = Slider(ax_slider_dataset, 'Dataset Index', 0, len(datasets) - 1, valinit=0, valfmt='%i', color=slider_color)
    slider_batch = Slider(ax_slider_batch, 'Batch Index', 0, 1, valinit=0, valfmt='%i', color=slider_color)


    def dataset_slider_changed(val):
        global current_dataset, img_generator

        if current_dataset != int(val):
            current_dataset = int(val)
            img_generator = generator.DiscreteFieldBoundaryGenerator(args.dataset, groups=[datasets[int(val)]], width=model.input.shape.as_list()[-2], height=model.input.shape.as_list()[-3],
                                                                     batch_size=grid_dim * grid_dim, shuffle=False, grayscale=model.input.shape.as_list()[-1] == 1,
                                                                     augmentation=False, distributed_augmentation=False, ooid=True)

            fig.suptitle(f'{datasets[int(val)]}', fontsize=10, ha='center')
            slider_batch.valmax = len(img_generator) - 1
            slider_batch.ax.set_xlim(slider_batch.valmin, slider_batch.valmax)
            slider_batch.set_val(0)

    def image_slider_changed(val):
        x, y, images, labels = img_generator[int(val)]
        predictions = model.predict(x)

        for axes, ax_img, net_input, img, label, prediction in zip(grid, grid_axes, x, images, labels, predictions):
            axes.lines = []
            ax_img.set_data(img)
            label_utils.draw_label(axes, img, label_utils.get_column_wise_y(label, img.shape[1]))
            spots = np.concatenate([np.linspace(0, 1, num=len(prediction))[:, np.newaxis], prediction if len(prediction.shape) > 1 else prediction[:, np.newaxis]], axis=1)
            line_model = fitting.fit_model(spots, step=1)
            label_utils.draw_label(axes, img, label_utils.get_column_wise_y(line_model, img.shape[1]), color='b')
            label_utils.draw_prediction(axes, img, prediction)

        fig.canvas.draw()

    def arrow_key_image_control(event):
        ind_dataset, ind_batch = int(slider_dataset.val), int(slider_batch.val)
        if event.key in ['a', 'd']:
            slider_dataset.set_val(max(ind_dataset-1, 0)) if event.key == 'a' else slider_dataset.set_val(min(ind_dataset+1, int(slider_dataset.valmax)))
        elif event.key in ['left', 'right']:
            slider_batch.set_val(max(ind_batch-1, 0)) if event.key == 'left' else slider_batch.set_val(min(ind_batch+1, int(slider_batch.valmax)))

    slider_dataset.on_changed(dataset_slider_changed)
    slider_batch.on_changed(image_slider_changed)
    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)

    dataset_slider_changed(0)

    plt.show()
