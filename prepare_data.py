import cv2
import os
from skimage import feature, exposure
import json
import numpy as np
from detect_face import get_face, find_face_1
import dlib
from imutils import face_utils


IMAGE_TYPES = ['jpg', 'png', 'jpeg']
VIDEO_TYPES = ['mp4', 'avi']
BULK_DATA = []


RAW_DATA_PATH = "data/raw_data/"
PREPAPRED_DATA_PATH = "data/prepared_data/"

ONEHOT_LABEL_PATH = "onehot_label.txt"

BATCH_SIZE = 3000

MAX_HEIGHT = 600
MAX_WIDTH = 600


face_landmark_model_path = 'prebuilt_model/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(face_landmark_model_path)


def display(image):
    cv2.imshow('img', image)
    cv2.waitKey(0)


# One hot encoding
def one_hot_encoding(names):
    length = len(names)
    ohc = {}
    for i,name in enumerate(names):
        vector = []
        for j in range(length):
            if j==i:
                vector.append(1)
            else:
                vector.append(0)
        ohc[name] = vector
    return ohc


COUNTER = 1
def dump_data(data):
    global COUNTER
    file_name = str(COUNTER)+".npy"
    file_path = os.path.join(PREPAPRED_DATA_PATH, file_name)
    np.save(file_path, data)
    COUNTER += 1


def feature_extraction_by_video(video, label):
    count = 1
    while True:
        flag, image = video.read()
        flag, image = video.read()
        flag, image = video.read()
        if not flag:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # display(image)
        feature_extraction_by_image(image, label)
        print(count)
        count += 1


def feature_extraction_by_image(image, label):
    global BULK_DATA
    try:
        if image.shape[0] > MAX_HEIGHT and image.shape[1] > MAX_WIDTH:
            image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))

        image_1 = cv2.flip(image, 1)
        rectangles_1 = find_face_1(image_1)
        rectangles = find_face_1(image)
        for rect,r in zip(rectangles, rectangles_1):
            face_feature = predictor(image, rect)
            face_feature = face_utils.shape_to_np(face_feature)

            face_feature_1 = predictor(image_1, r)
            face_feature_1 = face_utils.shape_to_np(face_feature_1)

            if len(face_feature) > 0:
                BULK_DATA.append([face_feature, label])
            if len(face_feature_1) > 0:
                BULK_DATA.append([face_feature_1, label])

            if len(BULK_DATA) >= BATCH_SIZE:
                dump_data(BULK_DATA)
                BULK_DATA = []

    except Exception as e:
        print("Error: ",e)


def check_media(file):
    media_format = file.split(".")[-1]
    if media_format in IMAGE_TYPES:
        return 1
    elif media_format in VIDEO_TYPES:
        return 2
    else:
        return 0


def prepare():
    global BULK_DATA
    labels = os.listdir(RAW_DATA_PATH)
    labels = one_hot_encoding(labels)
    with open(ONEHOT_LABEL_PATH, "w") as f:
        f.write(json.dumps(labels))
    for i, (key, label) in enumerate(labels.items(), 1):
        print("Folder: ",key)
        label = np.asarray(label)
        label_path = os.path.join(RAW_DATA_PATH, key)
        files = os.listdir(label_path)
        for j, file in enumerate(files):
            file_path = os.path.join(label_path, file)

            type = check_media(file)
            if type == 1:
                image = cv2.imread(file_path, 0)
                feature_extraction_by_image(image, label)

            elif type == 2:
                print("Processing Video: ", file_path)
                video = cv2.VideoCapture(file_path)
                feature_extraction_by_video(video, label)

            print(" %s  / %s"%(j, len(files)))


        if i == len(labels):
            dump_data(BULK_DATA)
            BULK_DATA = []


if __name__ == "__main__":
    prepare()