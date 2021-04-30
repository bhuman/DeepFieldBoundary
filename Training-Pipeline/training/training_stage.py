import callbacks
import color_conversion
import generator
import losses
import models
import multiprocessing
import numpy as np
import os
import tensorflow.keras as keras

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, TerminateOnNaN


def train(performance, directories, settings, checkpoint):
    # Deactivate GPU Usage if requested
    if not performance['use_cuda']:  os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Initiate the batch generators
    dataset_path = f'{directories["data_basedir"]}/{settings["dataset"]}'
    generator_class = eval(f'generator.{settings["generator"]}')
    train_generator = generator_class(dataset_path, settings['train_data'], width=settings['width'], height=settings['height'], batch_size=settings['batch_size'], shuffle=True, grayscale=settings['grayscale'], augmentation=settings['augmentation'], distributed_augmentation=performance['distributed_augmentation'])
    val_generator = generator_class(dataset_path, settings['val_data'], width=settings['width'], height=settings['height'], batch_size=settings['batch_size'], shuffle=False, grayscale=settings['grayscale'], augmentation=False, distributed_augmentation=performance['distributed_augmentation'])
    callbacks.TensorboardImageLogger(log_dir=f"{checkpoint['checkpoint_dir']}/tensorboard/input").log_images(train_generator[0][0] if settings['grayscale'] else np.asarray([color_conversion.yuv2rgb(img) for img in train_generator[0][0]]), name='Training first batch', epoch=checkpoint['epoch'])
    callbacks.TensorboardImageLogger(log_dir=f"{checkpoint['checkpoint_dir']}/tensorboard/input").log_images(val_generator[0][0] if settings['grayscale'] else np.asarray([color_conversion.yuv2rgb(img) for img in val_generator[0][0]]), name='Validation first batch', epoch=checkpoint['epoch'])

    # Initiate the loss and metric functions
    loss_function = eval(f'losses.{settings["loss"]}')
    metric_functions = [eval(f'losses.{_}') for _ in settings["metrics"]]

    # Initiate the model
    model_function = eval(f'models.{settings["model"]}')
    model = model_function(grayscale=settings['grayscale'], width=settings['width'], height=settings['height'], predict_uncertainty=settings['predict_uncertainty'], **settings['model_args'])
    model.compile(optimizer=settings['optimizer'], loss=loss_function, metrics=metric_functions)
    keras.utils.plot_model(model, to_file=f"{checkpoint['checkpoint_dir']}/{settings['model']}.png", show_shapes=True, show_layer_names=True, rankdir='TB')
    model.summary()

    # Define the callbacks
    train_callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0, patience=settings['es_patience'], verbose=0, mode='auto', restore_best_weights=False),
        ReduceLROnPlateau(monitor='val_loss', min_delta=1e-4, factor=0.25, patience=settings['lr_patience'], min_lr=1e-7),
        TensorBoard(log_dir=f"{checkpoint['checkpoint_dir']}/tensorboard", histogram_freq=1, write_graph=True),
        TerminateOnNaN(),

        callbacks.CustomModelCheckpoint(checkpoint=checkpoint)
    ]

    # Train the model
    model.fit(train_generator, validation_data=val_generator, initial_epoch=checkpoint['epoch'], epochs=settings['epochs'], callbacks=train_callbacks, shuffle=False, use_multiprocessing=performance['use_multiprocessing'], workers=eval(performance['workers']))
