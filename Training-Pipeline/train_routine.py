import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training"))

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from tensorboard import program
from training_stage import train


def start_train_routine(settings_file):
    with open(settings_file) as f:
        settings = json.load(f)

    # Initialize a new checkpoint directory
    checkpoint_dir = os.path.join(os.path.dirname(__file__), f"{settings['directories']['checkpoints_basedir']}", f"checkpoint-{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Initialize a new checkpoint file
    checkpoint = {'checkpoint_dir': checkpoint_dir, 'epoch': 0, 'best_model': '', 'last_model': '', 'last_weights': '', 'last_optimizer_weights': ''}
    with open(f'{checkpoint_dir}/checkpoint.json', 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=4)

    # Save the trainings settings file
    with open(f'{checkpoint_dir}/trainings-settings.json', 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)

    # Launch tensorboard
    tensorboard_dir = f"{checkpoint['checkpoint_dir']}/tensorboard"
    Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tensorboard_dir])
    tb.launch()

    train(settings['performance'], settings['directories'], settings['training'], checkpoint)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('settings', type=str, help='The settings file')
    args = parser.parse_args()

    start_train_routine(args.settings)
