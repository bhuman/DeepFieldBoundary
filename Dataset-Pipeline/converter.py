import argparse
import csv
import cv2
import h5py
import json
import numpy as np
import os
import random
import sys

from pathlib import Path
from tqdm import tqdm


def load_synthetic_data(image_paths, mask_paths):
    masks = [np.frombuffer(open(file, 'rb').read(), dtype=np.uint8) for file in mask_paths]
    if len(background_images) > 0:
        images = []
        for image_path, mask in zip(image_paths, masks):
            image, mask = cv2.imread(image_path), np.asarray([[int(value) for value in row.split()] for row in mask.readlines()[3:]])
            background = cv2.resize(cv2.imread(str(random.choice(background_images))), (image.shape[1], image.shape[0]))
            replace_mask = np.logical_or(mask == 0, mask == 2)
            image[replace_mask] = background[replace_mask]
            images.append(cv2.imencode('.png', image)[1])
    else:
        images = [np.frombuffer(open(file, 'rb').read(), dtype=np.uint8) for file in image_paths]
    return images, masks


def convert_to_hdf5(base_directory, override):
    batch_size = 1000

    for labels_filepath in sorted(Path(base_directory).rglob('*labels.csv')):
        print(f'Processing {labels_filepath}...')
        basedir = os.path.dirname(labels_filepath)
        dataset_name = os.path.relpath(basedir, base_directory)
        dataset_path = f'{base_directory}/{dataset_name.replace("/", "_")}.hdf5'

        if os.path.isfile(dataset_path):
            if override:
                os.remove(dataset_path)
            else:
                print(f'Dataset already exists, skipping {dataset_name}... \n')
                continue

        dataset = h5py.File(dataset_path, 'a')
        with open(labels_filepath, newline='') as csv_file:
            csv_data = np.asarray(list(csv.reader(csv_file)))
        labels = csv_data[:, 1:5].astype(np.float)
        image_paths = np.asarray([f'{basedir}/{image_name}' for image_name in csv_data[:, 0]])
        mask_paths = np.asarray([f'{os.path.splitext(image_path)[0]}.pgm' for image_path in image_paths])
        load_masks = np.all([os.path.isfile(mask_path) for mask_path in mask_paths])

        with tqdm(total=image_paths.shape[0], file=sys.stdout, unit=' Images') as progress:
            dataset.create_dataset(f'{dataset_name}/labels', data=labels, maxshape=labels.shape, dtype=np.float)
            images_dataset = dataset.create_dataset(f'{dataset_name}/images', (image_paths.shape[0],), dtype=h5py.vlen_dtype(np.uint8))
            masks_dataset = dataset.create_dataset(f'{dataset_name}/masks', (mask_paths.shape[0],), dtype=h5py.vlen_dtype(np.uint8)) if load_masks else None
            if os.path.isfile(f'{basedir}/mapping.json'):
                dataset.create_dataset(f'{dataset_name}/mapping', data=json.dumps(json.loads(open(f'{basedir}/mapping.json', 'r+').read()), indent=4))

            for index in range(0, image_paths.shape[0], batch_size):
                if load_masks:
                    images, masks = load_synthetic_data(image_paths[index:index + batch_size], mask_paths[index:index + batch_size])
                    images_dataset[index:index + batch_size] = images
                    masks_dataset[index:index + batch_size] = masks
                else:
                    images_dataset[index:index + batch_size] = [np.frombuffer(open(file, 'rb').read(), dtype=np.uint8) for file in image_paths[index:index + batch_size]]
                progress.update(image_paths[index:index + batch_size].shape[0])

        dataset.flush()
        dataset.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_directory', type=str)
    parser.add_argument('-b', '--background_directory', type=str, nargs='?', help='Directory containing background images for synthetic images')
    parser.add_argument('-o', '--override', action='store_true', help='Whether to override existing groups')
    args = parser.parse_args()

    background_images = list(Path(args.background_directory).rglob('*.jpg')) if args.background_directory else []
    convert_to_hdf5(args.base_directory, args.override)
