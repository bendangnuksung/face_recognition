import cv2
import face_recognition
import pickle
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--data_path", default="data/raw_data/", help="path of the data folder")
args = vars(ap.parse_args())

prepared_data_path = "data/prepared_data/prepare_data.json"


image_types = ["jpg", "jpeg", "png", "tif"]
video_types = ["mp4", "mpeg", "avi", "flv", "wmv", "mov"]


VIDEO_FRAME_SKIP_COUNTER = 100


def dump_serialize_encodings(data):
    """
    Dump Encodings to 'prepared_data_path'
    :param data:
    :return:
    """
    with open(prepared_data_path, "wb") as f:
        f.write(pickle.dumps(data))


def get_media_type(file):
    """
    Checks if file type is:
    1) Image
    2) Video
    :param file:
    :return:
    """
    file = file.split("/")[-1]
    file_type = file.split(".")[-1]

    if file_type in image_types:
        return "image"
    elif file_type in video_types:
        return "video"
    else:
        return None


def detect_face(image, train=False):
    """
    Given an image searches for face
    :param image:
    :param train:
    :return: coordinates list of the faces [( X1, Y2, X2, Y1 ) ,.... ]- format
    """
    if train:
        method = "cnn"
    else:
        method = "hog"

    bounding_boxes = face_recognition.face_locations(image, model=method)
    return bounding_boxes


def encode_face(image, bounding_boxes):
    """
    Given an image and the coordinates gets 128 features
    :param image:
    :param bounding_boxes:
    :return: 128 features of each cooridinates
    """
    encodings = face_recognition.face_encodings(image, bounding_boxes)
    return encodings


def encode_by_image(image, is_detect=False):
    if is_detect:
        bounding_boxes = detect_face(image)
    else:
        bounding_boxes = [(0, image.shape[1], image.shape[0], 0)]

    if len(bounding_boxes) == 1:
        # An image should contain only one FACE
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        encoding = encode_face(image, bounding_boxes)
        return encoding
    else:
        return None


def encode_by_video(video, skips=VIDEO_FRAME_SKIP_COUNTER):
    encodings = []
    while True:
        for i in range(skips):
            _, _ = video.read()
        flag, image = video.read()

        if not flag:
            break

        encoding = encode_by_image(image, is_detect=True)
        if encoding != None:
            encodings += encoding

    return encodings


def prepare_data():
    folders = os.listdir(args["data_path"])

    names = []
    encodings = []

    for folder in folders:
        name = folder
        folder_path = os.path.join(args["data_path"], folder)
        files = os.listdir(folder_path)
        print("Reading from ", folder_path)

        for i, file in enumerate(files, 1):
            file_path = os.path.join(folder_path, file)
            file_type = get_media_type(file)

            if file_type == "image":
                try:
                    image = cv2.imread(file_path, 0)
                    encode_value = encode_by_image(image)
                    if encode_value != None:
                        names.append(name)
                        encodings.append(encode_value)
                except Exception as e:
                    print("Error at Image Encoding:", e)

            elif file_type == "video":
                try :
                    video = cv2.VideoCapture(file_path)
                    encoded_values = encode_by_video(video)

                    for encoded_value in encoded_values:
                        names.append(name)
                        encodings.append(encoded_value)
                except Exception as e:
                    print("Error at Video Encoding:", e)

            print("%s / %s" % (i, len(files)))

    name_encoding_dict = {"names": names, "encodings": encodings}
    dump_serialize_encodings(name_encoding_dict)


if __name__ == "__main__":
    prepare_data()