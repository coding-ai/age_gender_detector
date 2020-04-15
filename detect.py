import argparse
import os
import cv2
import numpy as np
import time

class AgeGenderDetector():

    def __init__(self,args):
        self.face = args['face']
        self.image = args['image']
        self.age = args['age']
        self.gender = args['gender']
        self.confidence = args['confidence']

        # define the list of age buckets and genders our age/gender detector will predict
        self.AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.GENDER_LIST = ['Male', 'Female']

    def detect(self):
        # load serialized face detector model from disk
        print('[INFO] loading face detector model...')
        face_prototxt = os.path.sep.join([self.face, "deploy.prototxt"])
        face_weights = os.path.sep.join([self.face, "res10_300x300_ssd_iter_140000.caffemodel"])
        faceNet = cv2.dnn.readNet(face_prototxt, face_weights)

        # load serialized age detector model from disk
        print('[INFO] loading age detector model...')
        age_prototxt = os.path.sep.join([self.age, "age_deploy.prototxt"])
        age_weights = os.path.sep.join([self.age, "age_net.caffemodel"])
        ageNet = cv2.dnn.readNet(age_prototxt, age_weights)

        # load serialized gender detector model from disk
        print('[INFO] loading gender detector model...')
        gender_prototxt = os.path.sep.join([self.gender, "gender_deploy.prototxt"])
        gender_weights = os.path.sep.join([self.gender, "gender_net.caffemodel"])
        genderNet = cv2.dnn.readNet(gender_prototxt, gender_weights)

        # load input image and construct an input blob for the image
        print("[INFO] loading image...")
        image = cv2.imread(self.image)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0,  (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        print("[INFO] computing face detections...")
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
	        # extract the confidence (i.e., probability) associated with the prediction
	        confidence = detections[0, 0, i, 2]
            
	        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
	        if confidence > self.confidence:
		        # compute the (x, y)-coordinates of the bounding box for the object
		        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		        (startX, startY, endX, endY) = box.astype("int")

		        # extract the ROI of the face and then construct a blob from *only* the face ROI
		        face = image[startY:endY, startX:endX]
		        faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

		        # make predictions on the age and find the age bucket with the largest corresponding probability
		        ageNet.setInput(faceBlob)
		        preds = ageNet.forward()
		        i = preds[0].argmax()
		        age = self.AGE_LIST[i]
		        ageConfidence = preds[0][i]

                # make predictions on the gender and find the gender with the largest corresponding probability
		        genderNet.setInput(faceBlob)
		        preds = genderNet.forward()
		        i = preds[0].argmax()
		        gender = self.GENDER_LIST[i]
		        genderConfidence = preds[0][i]

		        # display the predicted age and gender to our terminal
		        text = "{}: {:.2f}%, {}: {:.2f}%".format(gender, genderConfidence * 100,age, ageConfidence * 100)
		        print("[INFO] {}".format(text))

		        # draw the bounding box of the face along with the associated predicted age and gender
		        y = startY - 10 if startY - 10 > 10 else startY + 10
		        cv2.rectangle(image, (startX, startY), (endX, endY),
			        (128, 191, 255), 6)
		        cv2.putText(image, text, (startX, y),
			        cv2.FONT_HERSHEY_TRIPLEX, 1, (128, 191, 255), 2)


        # save the output image
        cv2.imwrite('output.jpg',image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        help='path to input image'
    )
    parser.add_argument(
        '-f',
        '--face',
        help='path to face detector model directory'
    )
    parser.add_argument(
        '-a',
        '--age',
        help='path to age detector model directory'
    )
    parser.add_argument(
        '-g',
        '--gender',
        help='path to gender detector model directory'
    )
    parser.add_argument(
        '-c',
        '--confidence',
        type=float,
        default=0.5,
        help='minimum probability to filter weak detections'
    )

    args = vars(parser.parse_args())

    detector = AgeGenderDetector(args)
    detector.detect()