import cv2
import imgaug.augmenters as iaa
import multiprocessing
import numpy as np

from imgaug.augmentables import Keypoint, KeypointsOnImage
from joblib import Parallel, delayed


def random_shape_func_images(images, random_state, parents, hooks):
    result = []
    for i, image in enumerate(images):
        img_lights = np.zeros(image.shape, dtype=np.float)
        for j in range(np.random.randint(3,10)):
            img_light = np.zeros(image.shape, dtype=np.uint8)
            pts = np.random.randint(0, max(image.shape), (np.random.randint(4, 6), 2))
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(img_light, [pts], (1, 1, 1))
            img_lights = img_lights + img_light.astype(np.float) * np.random.rand(1)
        # img_lights_normalized = img_lights / img_lights.max()
        result.append(np.clip(image.astype(np.float) * (img_lights + 1), 0, 255).astype(np.uint8))
    return result


def random_shape_func_heatmaps(heatmaps, random_state, parents, hooks):
    return heatmaps


def random_shape_func_keypoints(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images


drop_light_shadow_generator = iaa.Lambda(
    func_images=random_shape_func_images,
    func_heatmaps=random_shape_func_heatmaps,
    func_keypoints=random_shape_func_keypoints
)


def augmentation(images, annotations, distributed=False):
    height, width, _ = images[0].shape
    keypoints = [KeypointsOnImage(
        [
            Keypoint(x=0, y=annotation[0]*height),
            Keypoint(x=annotation[1]*width, y=annotation[2]*height),
            Keypoint(x=width, y=annotation[3]*height)
        ], shape=(height, width)) for annotation in annotations]

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.125))),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5),
            iaa.Sometimes(0.5, drop_light_shadow_generator),
            iaa.SomeOf((0, 3), [
                iaa.Multiply((0.75, 1.5), per_channel=False),
                iaa.BlendAlphaFrequencyNoise(
                    exponent=(-1, 1),
                    foreground=iaa.Multiply((0.7, 1.2)),
                    background=iaa.LinearContrast((0.75, 1.5))
                ),
                iaa.MotionBlur(k=[3, 9]),
                iaa.Add((-20, 20), per_channel=0.5),
                iaa.LinearContrast((0.75, 1.5), per_channel=0.5)
            ], random_order=True)
        ], random_order=False).to_deterministic()

    if distributed:
        data = np.asarray(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(seq)(image=img, keypoints=kps) for img, kps in zip(images, keypoints)), dtype=object)
        augmented_images, augmented_keypoints = data[:, 0], data[:, 1]
    else:
        augmented_images, augmented_keypoints = seq(images=images, keypoints=keypoints)

    augmented_annotations = []
    for i, k in enumerate(augmented_keypoints):
        if k[0].x > k[2].x:  k = k[::-1]

        peak = (-1, -1)
        if annotations[i][1] == -1 and annotations[i][2] == -1:
            x, y = [k[0].x, k[2].x], [k[0].y, k[2].y]
        elif k[1].x < 0 or (k[0].y < 0 and k[1].y < 0) or (k[0].y > height and k[1].y > height):
            x, y = [k[1].x, k[2].x], [k[1].y, k[2].y]
        elif k[1].x > width or (k[1].y < 0 and k[2].y < 0) or (k[1].y > height and k[2].y > height):
            x, y = [k[0].x, k[1].x], [k[0].y, k[1].y]
        else:
            x, y = [k[0].x, k[1].x, k[2].x], [k[0].y, k[1].y, k[2].y]
            peak = (x[1]/width, np.interp(x[1], x, y)/height)
        augmented_annotation = [np.interp(0, x, y)/height, peak[0], peak[1], np.interp(width, x, y)/height]

        if augmented_annotation[0] < 0 and augmented_annotation[3] < 0:
            augmented_annotation = [0, -1, -1, 0]
        elif augmented_annotation[0] > 1 and augmented_annotation[2] > 1 and augmented_annotation[3] > 1:
            augmented_annotation = [1, -1, -1, 1]

        augmented_annotations.append(augmented_annotation)
    return augmented_images, np.asarray(augmented_annotations)
