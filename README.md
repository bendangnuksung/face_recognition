# Face Recognition

Face recognition with Python3 using OpenSource library [face_recognition](https://pypi.org/project/face_recognition/)

A simple goto Face Recognition

![demo](https://github.com/bendangnuksung/face_detection/blob/master/demo.gif)

## Pre Requirements
It accepts both images and videos for training. So you can just: 

* Create directory and put all images and video belonging to a particular person
* Rename directory to the respective person name
* Repeat with the remaining people

* Note:
  * An image or a video should contain only the face of the respective person
  * Minimum 15 images of every individual person (for better prediction)
  * Optional: Move all directories with images to data/raw_data/

## Using Face Recognition

Install dependencies

```sh
sudo apt-get install build-essential cmake pkg-config
# Dlib installation will take time (Building from src).
sudo pip3 install -r requirements.txt 
```

Now Prepare Data (without GPU it took me 20 min to encode 3000 images)

```sh
# if all directories with images has been moved to data/raw_data/
python3 prepare_data.py

# Otherwise
python3 prepare_data.py -p PATH_TO_DATA_DIRECTORIES 
```

Once completed. We can start testing

```sh
# face_recognition using image
python3 predict_face.py -i IMAGE_PATH

# face_recognition using video
python3 predict_face.py -v VIDEO_PATH
``` 

## Credits
[ageitgey](https://github.com/ageitgey/face_recognition)