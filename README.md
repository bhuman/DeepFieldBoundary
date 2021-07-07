# DeepFieldBoundary

This is the code from the paper "Soccer Field Boundary Detection Using Convolutional Neural Networks".

## Environment

The directory `Environment` contains a script to build and launch a docker image.

## Dataset

The dataset is located at https://sibylle.informatik.uni-bremen.de/public/datasets/fieldboundary. It can be downloaded by running `python downloader.py` within the `Dataset-Pipeline` directory.

To label new data, `Dataset-Pipeline/labeltool.py` can be used. It expects the path to a directory with PNG images as argument and creates the file `labels.csv` in that directory. See the comment in `Dataset-Pipeline/labeltool.py` for further instructions. The labeled directory must be converted to HDF5 using `Dataset-Pipeline/converter.py`. The argument must be the directory which contains the image directory (the idea is to have a directory tree with image directories which is transformed into a set of HDF5 files). Finally, the script `Dataset-Pipeline/dataset/linker.py` creates a single HDF5 file which links to the original datasets.

The dataset can be inspected using `hdf5-viewer.py` (e.g. `python hdf5-viewer.py Dataset-Pipeline/datasets/fieldboundary.hdf5`).

## Training

Training is configured in JSON files in `Training-Pipeline/settings`. The script `Training-Pipeline/train_routine.py` with a settings file as argument performs the training. This will create a directory in `Training-Pipeline/checkpoints` which contains the trained models per epoch and metadata. This directory can afterwards be passed to the `Training-Pipeline/evaluate.py` to generate numbers. `Training-Pipeline/test_routine.py` with a specific model allows to visually inspect the results of the model.
