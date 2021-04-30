import sys
sys.path.append("../Utils")

import argparse
import csv
import image_loading
import numpy as np
import os

from pathlib import Path


def exclude_image(dir, image):
    Path(f'{dir}/excluded').mkdir(parents=True, exist_ok=True)
    for file in sorted(Path(dir).glob(f'{image}*')):
        os.rename(file, f'{dir}/excluded/{os.path.basename(file)}')


def calc_intersect(p11, p12, p21, p22):
    s = np.vstack([p11, p12, p21, p22])
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2)
    if z == 0:
        return (np.asarray(p11) + np.asarray(p21)) * 0.5
    return np.asarray((x/z, y/z))


def rotate_points(points, left, right):
    dif = np.array(right) - np.array(left)
    angle = np.arctan2(dif[1], dif[0])
    sin, cos = np.sin(-angle), np.cos(-angle)
    rotated_points = np.asarray([(p[0]*cos - p[1]*sin, p[0]*sin + p[1]*cos) for p in points])
    return rotated_points


def get_peak_index(points, horizontal_check_point, eps, default_index):
    if np.all(abs(points[:, 1]-horizontal_check_point[1]) < eps):
        return default_index
    return min([(v, i) for (i, v) in enumerate(points)], key=lambda p: p[0][1])[1]


def side_checker(first, second):
    m = (first[1]-second[1]) / (first[0]-second[0])
    b = (first[0]*second[1] - second[0]*first[1]) / (first[0]-second[0])
    line_points = [[x, int(m*x+b)] for x in range(int(first[0]), int(second[0]))]
    return (np.all(list(np.all(mask[:max(0, p[1]-5), p[0]] != 1) for p in line_points)) and
            np.all(list(np.all(mask[max(0, p[1]+5):, p[0]] != 0) for p in line_points)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script converts UERoboCup segmentation masks to field boundary labels.')
    parser.add_argument('directory', nargs='?', default='.')
    args = parser.parse_args()
    labels_dict = {}

    for mask_filepath in sorted(Path(args.directory).glob('*.pgm')):
        mask = image_loading.load_pgm(str(mask_filepath))
        boundary = list((mask.shape[0] - (np.any(list((mask == x) for x in [0, 2]), axis=0))[::-1, :].argmax(axis=0)) % mask.shape[0])
        boundary = np.asarray([boundary[col] if ((boundary[col] < mask.shape[0]-1) and (mask[boundary[col]+1, col] == 1)) else -1 for col in range(mask.shape[1])])
        valid_boundary_indices = np.where(boundary > 0)[0]

        points = np.asarray(sorted(list(zip(valid_boundary_indices, boundary[valid_boundary_indices]))))
        if points.shape[0] < 2:
            if (np.count_nonzero(mask == 1) / np.size(mask) > 0.25) and side_checker([0, 0], [mask.shape[1], 0]):
                labels_dict[f'{os.path.splitext(os.path.basename(mask_filepath))[0]}.png'] = (0, -1, -1, 0)
            else:
                exclude_image(os.path.dirname(mask_filepath), os.path.splitext(os.path.basename(mask_filepath))[0])
            continue

        label = [-1, -1, -1, -1]
        left_border, right_border = points[0], points[-1]
        rotated_points = rotate_points(points, left_border, right_border)
        peak_index = get_peak_index(rotated_points, rotated_points[0], 1, -1)
        if peak_index < 0:
            peak = (np.asarray(left_border) + np.asarray(right_border)) * 0.5
        else:
            peak = points[peak_index]

            left_points = points[:peak_index+1]
            rotated_points_left = rotate_points(left_points, left_border, peak)
            left_support = left_points[get_peak_index(rotated_points_left, rotated_points_left[0], 2, peak_index)]

            right_points = points[peak_index:]
            rotated_points_right = rotate_points(right_points, peak, right_border)
            right_support = right_points[get_peak_index(rotated_points_right, rotated_points_right[0], 2, 0)]

            peak = calc_intersect(left_border, left_support, right_support, right_border)
            label[1:3] = [peak[0] / mask.shape[1], peak[1] / mask.shape[0]]
        left_border = calc_intersect(left_border, peak, (0, 0), (0, mask.shape[0]))
        right_border = calc_intersect(right_border, peak, (mask.shape[1], 0), (mask.shape[1], mask.shape[0]))

        if (not side_checker(left_border, peak)) or (not side_checker(peak, right_border)):
            exclude_image(os.path.dirname(mask_filepath), os.path.splitext(os.path.basename(mask_filepath))[0])
            continue
        label[0::3] = [left_border[1] / mask.shape[0], right_border[1] / mask.shape[0]]
        labels_dict[f'{os.path.splitext(os.path.basename(mask_filepath))[0]}.png'] = tuple(label)

    with open(f'{os.path.dirname(mask_filepath)}/labels.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows([(image, y1, x2, y2, y3) for image, (y1, x2, y2, y3) in labels_dict.items()])
