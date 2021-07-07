import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import sys
sys.path.append(os.path.join(os.path.dirname(__file_), "..", "Utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training"))

import fitting
import generator
import json
import label_utils
import numpy as np

from argparse import ArgumentParser
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model


def evaluate_checkpoint(checkpoint_path):
    with open(f'{checkpoint_path}/checkpoint.json') as f:
        checkpoint = json.load(f)
    with open(f'{checkpoint_path}/trainings-settings.json') as f:
        trainings_settings = json.load(f)
        directories, settings = trainings_settings['directories'], trainings_settings['training']

    model = load_model(f"{checkpoint_path}/{checkpoint['best_model'][checkpoint['best_model'].index('models'):]}", compile=False)
    img_generator = generator.DiscreteFieldBoundaryGenerator(f"{directories['data_basedir']}/{settings['dataset']}",
                                                             groups=settings['val_data'],
                                                             width=model.input.shape.as_list()[-2],
                                                             height=model.input.shape.as_list()[-3],
                                                             batch_size=settings['batch_size'], shuffle=False,
                                                             grayscale=settings['grayscale'],
                                                             augmentation=False, distributed_augmentation=False,
                                                             ooid=False)
    y_true = np.asarray([[label_utils.get_field_boundary_y(sample['label'], (0.5 + j) / img_generator.width) for j in
                          range(img_generator.width)] for sample in img_generator.samples])

    predictions = model.predict(img_generator, verbose=1)
    has_uncertainty = len(predictions.shape) > 2
    predictions = np.atleast_3d(predictions)

    fitted_predictions = []
    fitted_predictions_unit_weight = []
    for pred in predictions:
        spots = np.concatenate([np.linspace(0, 1, num=len(pred))[:, np.newaxis], pred], axis=1)
        fitted_model = fitting.fit_model(spots, step=1)
        fitted_predictions.append([label_utils.get_field_boundary_y(fitted_model, (0.5 + j) / img_generator.width) for j in range(img_generator.width)])
        if has_uncertainty:
            spots = np.concatenate([np.linspace(0, 1, num=len(pred))[:, np.newaxis], pred[:, :1]], axis=1)
            fitted_model = fitting.fit_model(spots, step=1)
            fitted_predictions_unit_weight.append([label_utils.get_field_boundary_y(fitted_model, (0.5 + j) / img_generator.width) for j in range(img_generator.width)])
    fitted_predictions = np.asarray(fitted_predictions)
    if has_uncertainty:
        fitted_predictions_unit_weight = np.asarray(fitted_predictions_unit_weight)

    print(f'MAE: {mean_absolute_error(y_true, predictions[..., 0])}')
    print(f'MAE fitted: {mean_absolute_error(y_true, fitted_predictions)}')
    if has_uncertainty:
        print(f'MAE fitted with unit weights: {mean_absolute_error(y_true, fitted_predictions_unit_weight)}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Checkpoint to evaluate')
    args = parser.parse_args()

    evaluate_checkpoint(checkpoint_path=args.checkpoint)
