import setting.constant as const
import numpy as np
import cv2

def overlay(image, layer):
    if (len(layer.shape) == 2):
        layer = cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    layer = cv2.cvtColor(layer, cv2.COLOR_BGR2BGRA)

    layer[np.where((layer == [0,0,0,255]).all(axis=2))] = const.BACKGROUND_COLOR + [255]
    layer[np.where((layer == [255,255,255,255]).all(axis=2))] = const.SEGMENTATION_COLOR + [255]
    layer = cv2.addWeighted(image, 0.6, layer, 0.4, 0)
    return layer

def light(image, bright, contrast):
    bright = bright * 1.2
    contrast = contrast * 2
    image = image * ((contrast/127)+1) - contrast + bright
    image = np.clip(image, 0, 255)
    return np.uint8(image)

def threshold(image, min_limit=None, max_limit=255, clip=0):
    if min_limit is None:
        min_limit = int(np.mean(image) + clip)

    _, image = cv2.threshold(image, min_limit, max_limit, cv2.THRESH_BINARY)
    return np.uint8(image)

def gauss_filter(image, kernel=(3,3), iterations=1):
    for _ in range(iterations):
        image = cv2.GaussianBlur(image, kernel, 0)
    return np.uint8(image)

def median_filter(image, kernel=3, iterations=1):
    for _ in range(iterations):
        image = cv2.medianBlur(image, kernel, 0)
    return np.uint8(image)

def equalize_light(image, limit=3, grid=(7,7), gray=False):
    if (len(image.shape) == 2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = True
    
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))

    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if gray: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.uint8(image)

def back_in_black(image):
    image = light(image.copy(), bright=120, contrast=60)
    black_level = 0

    for x in range(6):
        bi = threshold(image, clip=-x)
        if (bi==0).sum() > (bi==255).sum():
            black_level += 1

    return (black_level > 3)


######### Unused #########


def edges(image, threshold1=250, threshold2=350, kernel=3):
    image = cv2.Canny(image, threshold1, threshold2, kernel)
    image = cv2.bitwise_not(image)
    return np.uint8(image)

def equalize_hist(image):
    image = cv2.equalizeHist(image)
    return np.uint8(image)

def otsu(img):
    hist = np.zeros(256, dtype=int)

    for y in range(len(img)):
        for x in range(len(img[0])):
            hist[int(img[y,x])] += 1

    total = (len(img) * len(img[0]))

    current_max, threshold = 0, 0
    sumT, sumF, sumB = 0, 0, 0

    weightB, weightF = 0, 0
    varBetween, meanB, meanF = 0, 0, 0

    for i in range(0,256):
        sumT += (i * hist[i])

    for i in range(0,256):
        weightB += hist[i]
        weightF = total - weightB
        if (weightF <= 0):
            break
        if (weightB <= 0):
            weightB = 1

        sumB += (i * hist[i])
        sumF = sumT - sumB
        meanB = sumB/weightB
        meanF = sumF/weightF
        varBetween = (weightB * weightF)
        varBetween *= (meanB-meanF) * (meanB-meanF)

        if (varBetween > current_max):
            current_max = varBetween
            threshold = i

    img[img <= threshold] = 0
    img[img > threshold] = 255
    return np.array(img, dtype=np.uint8)