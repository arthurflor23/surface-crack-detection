import dip.image as im
import numpy as np
import cv2

def image_preprocessor(image):
    image = im.equalize_light(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    black_level = im.back_in_black(image)

    image = im.gauss_filter(image, (3,3))
    image = im.light(image, bright=-30, contrast=-30)
    
    if not black_level:
        image = cv2.bitwise_not(image)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(image, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    image = np.subtract(image, mask)

    return im.threshold(image, clip=5)

def label_preprocessor(label):
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label = im.threshold(label, min_limit=127)
    return label

def posprocessor(image):
    # image = im.threshold(image)
    # kernel = np.ones((5,5), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)

    # image = im.median_filter(image, 11)
    # image = im.median_filter(image, 7, iterations=6)
    # image = im.median_filter(image, 5, iterations=6)

    # image = cv2.erode(image, kernel, iterations=1)
    # image = im.median_filter(image, 3, iterations=6)

    return im.threshold(image)