import sys
sys.path.append("../../Utils")

import argparse
import h5py
import hdf5_utils
import os

from glob import glob


def link(base_directory, dataset_name):
    if os.path.isfile(f'{dataset_name}.hdf5'):
        print(f'Dataset {dataset_name} already exists...')
        return

    dataset = h5py.File(f'{dataset_name}.hdf5', 'a')
    for hdf5_file_path in glob(f'{base_directory}/*.hdf5'):
        hdf5_file = h5py.File(hdf5_file_path, 'r')
        hdf5_file_datasets = [dataset.rsplit('/', 1) for dataset, key in hdf5_utils.h5py_dataset_iterator(hdf5_file)]
        for h5fd in hdf5_file_datasets:
            dataset[f'{h5fd[0]}/{h5fd[1]}'] = h5py.ExternalLink(hdf5_file_path, f'{h5fd[0]}/{h5fd[1]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_directory', type=str)
    parser.add_argument('dataset_name', nargs='?', default='fieldboundary', type=str)
    args = parser.parse_args()

    link(args.base_directory, args.dataset_name)
