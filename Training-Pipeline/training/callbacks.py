import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from pathlib import Path
from tensorflow.keras.models import save_model


class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, checkpoint):
        super(CustomModelCheckpoint, self).__init__()
        self.checkpoint = checkpoint
        self.best_val_loss = np.finfo(np.float).max

        # Create model save directory in case it got deleted
        self.model_dir = f"{self.checkpoint['checkpoint_dir']}/models/"
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        train_loss = round(logs.get('loss'), 4)
        val_loss = round(logs.get('val_loss'), 4)

        # Save model as h5 file
        model_filepath = f'{self.model_dir}/model-{epoch + 1}-{train_loss}-{val_loss}.h5'
        save_model(self.model, model_filepath)

        # Save model weights as h5 file
        weights_filepath = f'{self.model_dir}/model-weights-{epoch + 1}-{train_loss}-{val_loss}.h5'
        self.model.save_weights(weights_filepath)

        # Save optimizer state as pickle file
        optimizer_weights_filepath = f'{self.model_dir}/optimizer-weights-{epoch + 1}-{train_loss}-{val_loss}.pkl'
        with open(optimizer_weights_filepath, 'wb') as f:
            pickle.dump(K.batch_get_value(getattr(self.model.optimizer, 'weights')), f)

        # Update the checkpoint epoch
        self.checkpoint['epoch'] += 1
        if val_loss < self.best_val_loss:
            self.checkpoint['best_model'] = model_filepath
            self.best_val_loss = val_loss
        self.checkpoint['last_model'] = model_filepath
        self.checkpoint['last_weights'] = weights_filepath
        self.checkpoint['last_optimizer_weights'] = optimizer_weights_filepath
        with open(f"{self.checkpoint['checkpoint_dir']}/checkpoint.json", "w") as f:
            json.dump(self.checkpoint, f, indent=4)


class TensorboardImageLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    @staticmethod
    def plot_to_image():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=3)
        image = tf.expand_dims(image, 0)
        buf.close()
        return image

    def log_plot(self, name, epoch):
        file_writer = tf.summary.create_file_writer(self.log_dir)
        with file_writer.as_default():
            tf.summary.image(name, self.plot_to_image(), step=epoch)
        file_writer.close()

    def log_images(self, images, name, epoch):
        file_writer = tf.summary.create_file_writer(self.log_dir)
        with file_writer.as_default():
            tf.summary.image(name, images, max_outputs=1 if len(images.shape) == 3 else images.shape[0], step=epoch)
        file_writer.close()


class StepDecay:
    def __init__(self, init_alpha=0.01, factor=0.25, drop_every=10):
        self.init_alpha = init_alpha
        self.factor = factor
        self.drop_every = drop_every

    def __call__(self, epoch):
        exp = np.floor((1 + epoch) / self.drop_every)
        alpha = self.init_alpha * (self.factor ** exp)
        return float(alpha)


class PolynomialDecay:
    def __init__(self, max_epochs=100, init_alpha=0.01, power=1.0):
        self.max_epochs = max_epochs
        self.init_alpha = init_alpha
        self.power = power

    def __call__(self, epoch):
        decay = (1 - (epoch / float(self.max_epochs))) ** self.power
        alpha = self.init_alpha * decay
        return float(alpha)
