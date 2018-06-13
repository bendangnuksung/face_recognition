import cv2
import dlib


opencv_model_path = 'opencv_model/frontalface_default.xml'
detector_obj = cv2.CascadeClassifier(opencv_model_path)
detector = dlib.get_frontal_face_detector()


def find_face(image):
    face = detector_obj.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    return face


def find_face_1(image):
    face = detector(image, 1)
    return face


def get_face(image):
    positions = find_face_1(image)
    faces = []
    for position in positions:
        left = position.left()
        top = position.top()
        width = position.right() - left
        height = position.bottom() - top
        faces.append(image[top:top + height, left:left + width])
    return faces


def display(images):
    for img in images:
        cv2.imshow('img', img)
        cv2.waitKey(0)


def main(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = get_face(gray)
    display(faces)


if __name__ == "__main__":
    image_path = "data/raw_data/ben/ben1.jpg"
    image = cv2.imread(image_path)
    main(image)

