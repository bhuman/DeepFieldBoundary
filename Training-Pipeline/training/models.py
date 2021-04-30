from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Flatten, Input, MaxPooling2D, Reshape, SpatialDropout2D
from tensorflow.keras.models import Model


def conv_block(_x, filters, kernel_size, strides, padding, activation, name):
    x1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, name=name)(_x)
    x1 = BatchNormalization(name=f'{name}-batch-norm')(x1)
    return Activation(activation, name=f'{name}-activation')(x1)


def model1(grayscale=True, filters=16, width=40, height=30, predict_uncertainty=False, with_dropout=True, name='DeepFieldBoundary'):
    def stage(_x, _filters=16, _activation='relu', _name=None):
        x1 = conv_block(_x, filters=_filters // 4, kernel_size=1, strides=(1, 1), padding='valid', activation=_activation, name=f'{_name}-branch1-reduction')
        x2 = conv_block(x1, filters=_filters // 2, kernel_size=3, strides=(1, 1), padding='same', activation=_activation, name=f'{_name}-branch1-conv1')
        x3 = conv_block(x2, filters=_filters, kernel_size=3, strides=(2, 1), padding='same', activation=_activation, name=f'{_name}-branch1-conv2')

        y1 = conv_block(_x, filters=_filters // 2, kernel_size=1, strides=(1, 1), padding='valid', activation=_activation, name=f'{_name}-branch2-reduction')
        y2 = conv_block(y1, filters=_filters, kernel_size=3, strides=(2, 1), padding='same', activation=_activation, name=f'{_name}-branch2-conv')

        z1 = MaxPooling2D(3, strides=(2, 1), padding='same', name=f'{_name}-branch3-pool')(_x)
        z2 = conv_block(z1, filters=_filters, kernel_size=1, strides=(1, 1), padding='valid', activation=_activation, name=f'{_name}-branch3-reduction')

        return Concatenate(axis=-1, name=f'{_name}-concat')([x3, y2, z2])

    x = Input(shape=(height, width, 1 if grayscale else 3), name='input')
    inputs = x

    # 30x40x?
    x = stage(x, filters, _name='stage1')
    if with_dropout:
        x = SpatialDropout2D(0.25)(x)
    # 15x40x24

    # 15x40x24
    x = stage(x, filters, _name='stage2')
    if with_dropout:
        x = SpatialDropout2D(0.25)(x)
    # 8x40x24

    # 8x40x24
    x = stage(x, filters, _name='stage3')
    if with_dropout:
        x = SpatialDropout2D(0.25)(x)
    # 4x40x24

    # 4x40x24
    x = stage(x, filters, _name='stage4')
    # 2x40x24

    # 2x40x24
    x = Conv2D(2 if predict_uncertainty else 1, (2, 1), name='output-conv')(x)
    # 1x40x?

    # 1x40x?
    if predict_uncertainty:
        x = Reshape((width, 2), name='reshape')(x)
    else:
        x = Flatten(name='flatten')(x)
    # 40(x2)

    outputs = x
    return Model(inputs=inputs, outputs=outputs, name=name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='This script prints the summary of a model.')
    parser.add_argument('model', nargs='?', default='model1')
    args = parser.parse_args()

    model = eval(args.model)()
    model.summary()
