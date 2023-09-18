import cv2
import numpy as np


def preprocess(img, imgSize):
    # Your preprocessing code goes here
    # create target image and copy sample image into it
    widthTarget, heightTarget = imgSize
    height, width = img.shape
    factor_x = width / widthTarget
    factor_y = height / heightTarget

    factor = max(factor_x, factor_y)
    # scale according to factor
    newSize = (
        min(widthTarget, int(width / factor)),
        min(heightTarget, int(height / factor)),
    )

    img = cv2.resize(img, newSize)
    target = np.ones(shape=(heightTarget, widthTarget), dtype="uint8") * 255
    target[0 : newSize[1], 0 : newSize[0]] = img
    # transpose
    img = cv2.transpose(target)
    # standardization
    mean, stddev = cv2.meanStdDev(img)
    mean = mean[0][0]
    stddev = stddev[0][0]
    img = img - mean
    img = img // stddev if stddev > 0 else img
    return img


alphabets = "0123456789' "
max_str_len = 10  # max length of input labels
# My project have 7 digits per image, but as long as the max_str_len > the number of digit per image,
# it's just fine
num_of_characters = len(alphabets) + 1  # +1 for ctc pseudo blank
num_of_timestamps = 32  # max length of predicted labels
# I find out that if the num_of_timestamps ... I forgot it...


def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))

    return np.array(label_num)


def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret += alphabets[ch]
    return ret
