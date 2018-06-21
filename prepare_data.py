import cv2
import face_recognition
import pickle
import argparse
import os


# 1) Raw data path is where the folders of images, videos will be (Note: Folder Name = name of the person)
# 2) Prepared data path is where the encodings will be saved
raw_data_path = "data/raw_data/"
prepared_data_path = "data/prepared_data"
if not os.path.exists(prepared_data_path):
    os.makedirs(prepared_data_path)

# encoded file name -> file name of the trained encodings
encoded_file_name = "saved_encodings_1"
encoded_file_path = os.path.join(prepared_data_path, encoded_file_name)


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--data_path", default=raw_data_path, help="path of the data folder")
args = vars(ap.parse_args())

# video, image types to check file type
image_types = ["jpg", "jpeg", "png", "tif"]
video_types = ["mp4", "mpeg", "avi", "flv", "wmv", "mov"]


# Skips N frame in a video in each iteration
VIDEO_FRAME_SKIP_COUNTER = 150


IMAGE_PROCESSED = 0


def dump_serialize_encodings(data):
    """
    Dump Encodings to 'prepared_data_path'
    :param data:
    :return:
    """
    with open(encoded_file_path, "wb") as f:
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


def detect_face(image, method="hog"):
    """
    Given an image searches for face
    :param image:
    :param method: 1) cnn: more accurate but slow
                   2) hog: less accurate but fast
    :return: coordinates list of the faces [( X1, Y2, X2, Y1 ) ,.... ]- format where X1= Top, Y1= Left
    """
    if method not in ["cnn", "hog"]:
        print("Method not identified")
        exit()
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


def encode_by_image(image, is_detect=True):
    """
    Given an image, locates the face and get 128 features of face
    set is_detect = False --> set if given image is just the face
    :param image:
    :param is_detect:
    :return: lists of 128 features of face
    """
    global IMAGE_PROCESSED
    images = [image, cv2.flip(image, 1)] # Horizontal Flip image for better prediction
    encodings = []
    for image in images:
        if is_detect:
            bounding_boxes = detect_face(image)
        else:
            bounding_boxes = [(0, image.shape[1], image.shape[0], 0)]

        if len(bounding_boxes) == 1:
            # An image should contain only one FACE
            if len(image.shape) < 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            encoding = encode_face(image, bounding_boxes)
            encodings.append(encoding)
            IMAGE_PROCESSED += 1

    if len(encodings) > 0:
        print("Image processed: ", IMAGE_PROCESSED)
        return encodings
    return None


def encode_by_video(video, skips=VIDEO_FRAME_SKIP_COUNTER):
    """
    Takes Frames from image and feeded to "Encode_by_image()"
    :param video:
    :param skips: Skips 'N' frames, (To get more unique image rather than repetition"
    :return:
    """
    total_encodings = []
    while True:
        for i in range(skips):
            _, _ = video.read()
        flag, image = video.read()

        if not flag:
            break

        encodings = encode_by_image(image)
        if encodings is not None:
            for encoding in encodings:
                total_encodings += encoding

    return total_encodings


def prepare_data():
    folders = os.listdir(args["data_path"])

    names = []
    encodings = []

    for folder in folders:
        name = folder
        folder_path = os.path.join(args["data_path"], folder)

        if not os.path.isdir(folder_path):
            continue

        files = os.listdir(folder_path)
        print("Reading from ", folder_path)

        for i, file in enumerate(files, 1):
            file_path = os.path.join(folder_path, file)
            file_type = get_media_type(file)

            if file_type == "image":
                try:
                    image = cv2.imread(file_path, 0)
                    encoded_values = encode_by_image(image)
                    if encoded_values is not None:
                        for encoded_value in encoded_values:
                            names.append(name)
                            encodings += encoded_value
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

    name_encoding_dict = {"names": names, "encodings": encodings}
    dump_serialize_encodings(name_encoding_dict)


if __name__ == "__main__":
    prepare_data()