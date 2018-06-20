import cv2
import pickle
import face_recognition
import argparse
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to image for the prediction")
args = vars(ap.parse_args())

prepared_data_path = "data/prepared_data/prepare_data.dat"

MAX_HEIGHT = 1000
MAX_WIDTH = 1500

saved_encodings = None


def is_image_big(image):
    if image.shape[0] > MAX_HEIGHT and image.shape[1] > MAX_WIDTH:
        return True
    return False


def detect_face(image, method="hog"):
    """
    Given an image searches for face
    :param image:
    :param method: 1) cnn: more accurate but slow, good for training
                   2) hog: less accurate but fast, good for real time predicting
    :return: coordinates list of the faces [( X1, Y2, X2, Y1 ) ,.... ]- format
    """
    bounding_boxes = face_recognition.face_locations(image, model=method)
    return bounding_boxes


def get_encodings(image, bounding_boxes):
    """
    Given an image and the coordinates gets 128 features
    :param image:
    :param bounding_boxes:
    :return: 128 features of each cooridinates
    """
    encodings = face_recognition.face_encodings(image, bounding_boxes)
    return encodings


def load_encodings():
    """
    Restore Encodings from the saved directory
    :return:
    """
    global saved_encodings
    saved_encodings = pickle.loads(open(prepared_data_path, "rb").read())


def identify_person(face_encodings):
    """
    checks if given encodings matches with saved encodings
    :param face_encodings:
    :return: name of person if found
    """
    names = []
    for encoding in face_encodings:
        results = face_recognition.compare_faces(saved_encodings["encodings"], encoding)
        results = np.array(results)
        name = "Unidentified"

        indices = list(np.where(results)[0])
        if len(indices):
            result_names = [saved_encodings["names"][i] for i in indices]
            name = max(set(result_names), key=result_names.count)

        names.append(name)
    return names


def format_results(names, face_coordinates, reduced_flag):
    """
    Mapping names with the face coordinates,

    :param names:
    :param face_coordinates:
    :param reduced_flag: if reduced_flag == True : the original image has been shrunk by Half (/2),
                              restoring face coordinates to original size.
    :return: Dictionary {name: coordinates}
    """
    product = 2 if reduced_flag else 1
    name_coords = {}
    unknown_counter = 1
    for name, coords in zip(names, face_coordinates):
        x1 = coords[0] * product
        y2 = coords[1] * product
        x2 = coords[2] * product
        y1 = coords[3] * product

        if name == "Unidentified":
            name = name + "_" + unknown_counter
            name_coords[name] = [x1, y1, x2, y2]
            unknown_counter += 1
        else:
            name_coords[name] = [x1, y1, x2, y2]

    return name_coords


def predict(image):
    global saved_encodings
    reduced_flag = False

    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if is_image_big(image):
        image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
        reduced_flag = True
    if saved_encodings is None:
        load_encodings()

    face_coordinates = detect_face(image)
    face_encodings = get_encodings(image, face_coordinates)

    names = identify_person(face_encodings)
    result = format_results(names, face_coordinates, reduced_flag)
    return result


if __name__ == "__main__":
    image = cv2.imread(args["image"])
    print(predict(image))
