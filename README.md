# Age and Gender Classification

This code is based on the [OpenCV Age Detection with Deep Learning](https://www.pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/) by [Adrian Rosebrock](https://www.pyimagesearch.com/author/adrian/), which performs automatic age detection/prediction using OpenCV, Deep Learning and Python.

It has been refined and includes gender detection as extra feature. This code only works with static images.

The code follows a simple approach:

1. Detect faces un the input image/video stream
2. Extract the ROI and apply age and gender detector algorithm to predict the age and gender of the person.

Follow Adrian's tutorial to learn more about it.

## Requirements

To properly run this example you will need to have Python installed (Anaconda distribution) and OpenCV. The code uses pre-trained Caffe models, that you can find located in the folders `face_detector`, `age_detector`, `gender_detector`.

## Usage

1. Fork the repo.
2. Clone your forked repo.
3. Locate a picture on the images folder and run the following code:

Run `python detect.py --image images/[your_image_name] --face face_detector --age age_detector --gender gender_detector --confidence [confidence(0-1)]`

## Other

Most of the code is extracted directly from Adrian's repository. Only the gender detection was added as extra. If you have any questions regarding this particular case, you can contact me through my [email](codingartificalintelligence@gmail.com) or [instagram](https://www.instagram.com/ai.coding/).