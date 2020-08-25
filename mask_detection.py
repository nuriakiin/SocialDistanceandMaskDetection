import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import tensorflow as tf
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from mtcnn.mtcnn import MTCNN


def load_dataset():

    images = os.path.join("data/mask_detection_dataset/Medical Mask/Medical Mask/images")
    annotations = os.path.join("data/mask_detection_dataset/Medical Mask/Medical Mask/annotations")
    train_set = pd.read_csv(os.path.join("data/mask_detection_dataset/train.csv"))
    submission_set = pd.read_csv(os.path.join("data/mask_detection_dataset/submission.csv"))

    a = os.listdir(images)
    b = os.listdir(annotations)
    a.sort()
    b.sort()
    train_images = a[1698:]
    test_images = a[:1698]

    options = ['face_with_mask', 'face_no_mask']
    train_set = train_set[train_set['classname'].isin(options)]
    train_set.sort_values('name', axis=0, inplace=True)

    bbox = []
    for i in range(len(train_set)):
        arr = []
        for j in train_set.iloc[i][["x1", 'x2', 'y1', 'y2']]:
            arr.append(j)
        bbox.append(arr)
    train_set["bbox"] = bbox

    img_size = 50
    data = []
    path = '/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'

    for i in range(len(train_set)):
        arr = []
        for j in train_set.iloc[i]:
            arr.append(j)
        img_array = cv2.imread(os.path.join(images, arr[0]), cv2.IMREAD_GRAYSCALE)
        crop_image = img_array[arr[2]:arr[4], arr[1]:arr[3]]
        new_img_array = cv2.resize(crop_image, (img_size, img_size))
        data.append([new_img_array, arr[5]])



    return images, annotations, train_set, submission_set


def train(images, annotations, train_set, submission_set):

    return 0


def main():
    images, annotations, train_set, submission_set = load_dataset()
    print(len(train_set))
    train(images, annotations, train_set, submission_set)


if __name__ == "__main__":
    main()