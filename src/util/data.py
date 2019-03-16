from glob import glob
from util import path, misc
import setting.constant as const
import numpy as np
import cv2

def fetch_from_path(images):
    image_list = sorted(glob(path.join(images, const.FILTER)))
    image = np.array([cv2.imread(item, 1) for item in image_list])
    return image

def fetch_from_paths(images, labels):
    image_list, label_list = [], []

    for img, lab in zip(images, labels):
        img_fetch = sorted(glob(path.join(img, const.FILTER)))
        lab_fetch = sorted(glob(path.join(lab, const.FILTER)))

        img_fetch, lab_fetch = misc.shuffle(img_fetch, lab_fetch)

        for i, l in zip(img_fetch, lab_fetch):
            image_list.append(i)
            label_list.append(l)

    image = np.array([cv2.imread(item, 1) for item in image_list])
    label = np.array([cv2.imread(item, 1) for item in label_list])
    return image, label

def length_from_path(file_dir, *dirs):
    length_fetch = len(glob(path.join(file_dir, const.FILTER)))
    for x in dirs:
        length_fetch += len(glob(path.join(x, const.FILTER)))
    return length_fetch

def imshow(name, image):
    image = np.clip(image, 0, 255)
    cv2.imshow(name, np.uint8(image))
    cv2.waitKey(0)

def imwrite(file_name, image):
    image = np.clip(image, 0, 255)
    cv2.imwrite(file_name, np.uint8(image))