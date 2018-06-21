import cv2
import pickle
import face_recognition
import argparse
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to image for the prediction")
ap.add_argument("-v", "--video", help="Path to video")
ap.add_argument("-e", "--encode", help="Path saved encodings")
args = vars(ap.parse_args())

if args['encode'] is None:
    prepared_data_path = "data/prepared_data/saved_encodings_1"
else:
    prepared_data_path = args['encode']


MAX_HEIGHT = 800
MAX_WIDTH = 800

saved_encodings = None

# Name if face not Identified
UNKNOWN = "unknown"


def is_image_big(image):
    """
    Check if image height or width greater than MAX HEIGHT or MAX WIDTH
    :param image:
    :return:
    """
    if image.shape[0] > MAX_HEIGHT or image.shape[1] > MAX_WIDTH:
        return True
    return False


def resize_image(image):
    """
    Resizing image smaller for faster processing
    :param image:
    :return: resized_image, product (which can be multiplied to bring back to original shape)
    """
    height = image.shape[0]
    width = image.shape[1]

    if width > height:
        diff_product = MAX_WIDTH / width
        product = width / MAX_WIDTH
    else:
        diff_product = MAX_HEIGHT / height
        product = height / MAX_HEIGHT

    new_height = int(diff_product * height)
    new_width = int(diff_product * width)

    image = cv2.resize(image, (new_width, new_height))
    return image, product


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


def identify_person(face_encodings, threshold=10):
    """
    checks if given encodings matches with saved encodings
    :param face_encodings:
    :return: name of person if found
    """
    names = []
    for encoding in face_encodings:
        results = face_recognition.compare_faces(saved_encodings["encodings"], encoding)
        results = np.array(results)
        name = UNKNOWN

        indices = list(np.where(results)[0])
        if len(indices):
            result_names = {}
            for i in indices:
                temp_name = saved_encodings["names"][i]
                if temp_name in result_names:
                    result_names[temp_name] += 1
                else:
                    result_names[temp_name] = 1
            sorted_results = sorted(result_names.items(), key= lambda x: x[1], reverse=True)
            (result_name, count) = sorted_results[0]
            if count >= threshold:
                name = result_name
        names.append(name)
    return names


def format_results(names, face_coordinates, product):
    """
    Mapping names with the face coordinates,
    :param names:
    :param face_coordinates:
    :param diff_product: Product which will be mutiplied to bring back to original image shape
    :return: Dictionary {name: coordinates}
             Coordinates format [ x1, y1, x2, y2] where x1 = top, y1 = left
    """
    name_coords = {}
    unknown_counter = 1
    for name, coords in zip(names, face_coordinates):
        x1 = int(coords[0] * product)
        y2 = int(coords[1] * product)
        x2 = int(coords[2] * product)
        y1 = int(coords[3] * product)

        if name == UNKNOWN:
            name = name + "_" + str(unknown_counter)
            name_coords[name] = [x1, y1, x2, y2]
            unknown_counter += 1
        else:
            name_coords[name] = [x1, y1, x2, y2]

    return name_coords


def predict(image):
    """
    Locates multiple face and guesses name from an image
    :param image:q
    :return: dictionary of {name: coordinates}
             Coordinates format [ x1, y1, x2, y2] where x1 = top, y1 = left
    """
    global saved_encodings

    # Image needs to be in RGB to get encodings
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    product = 1
    if is_image_big(image):
        image, product = resize_image(image)

    # Loading saved faces encodings
    if saved_encodings is None:
        load_encodings()

    face_coordinates = detect_face(image)
    face_encodings = get_encodings(image, face_coordinates)

    names = identify_person(face_encodings)
    result = format_results(names, face_coordinates, product)
    return result


def add_name_to_image(image, name_coords):
    """
    Draw bounding box on face and add name text
    :param image:
    :param name_coords:
    :return:
    """
    if image.shape[0] > 2000 or image.shape[1] > 2000:
        rect_line_thickness = 5
        text_line_thickness = 3
        text_size = 4

    elif image.shape[0] > 1000 or image.shape[1] > 1000:
        rect_line_thickness = 3
        text_line_thickness = 2
        text_size = 2

    elif image.shape[0] > 5000 or image.shape[1] > 5000:
        rect_line_thickness = 2
        text_line_thickness = 2
        text_size = 1

    else:
        rect_line_thickness = 1
        text_line_thickness = 1
        text_size = 0.5

    for name, coords in name_coords.items():
        if UNKNOWN in name:
            cv2.rectangle(image, (coords[1], coords[0]), (coords[3], coords[2]), (0, 0, 255), rect_line_thickness)
            cv2.putText(image, name, (coords[1], coords[0]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), text_line_thickness)
        else:
            cv2.rectangle(image, (coords[1], coords[0]), (coords[3], coords[2]), (0, 255, 0), rect_line_thickness)
            cv2.putText(image, name, (coords[1], coords[0]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), text_line_thickness)

    return image


def process_image(image):
    name_coords = predict(image)
    image = add_name_to_image(image, name_coords)
    cv2.imshow('vid', image)
    cv2.waitKey(0)


def process_video(video, skips=15):
    """
    Predicts face from the video
    :param video: video object (cv2.VideoCapture)
    :param skips: Skips N frames in each iteration
    :return:
    """
    while True:
        for i in range(skips):
            flag, image = video.read()
        flag, image = video.read()

        if not flag:
            break

        name_coords = predict(image)
        if name_coords:
            image = add_name_to_image(image, name_coords)
        if is_image_big(image):
            image, _ = resize_image(image)
        cv2.imshow('vid', image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":

    if args["video"] is not None:
        video = cv2.VideoCapture(args["video"])
        process_video(video)

    elif args["image"] is not None:
        image = cv2.imread(args["image"])
        process_image(image)

    else:
        print("Please provide path to image or video:")
        print("-i image_path")
        print("-v video_path")