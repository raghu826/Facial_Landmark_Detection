# to run the file
# python Landmark_images.py --image images/1.jpg --shape-predictor shape_predictor_68_face_landmarks.dat

import dlib
import cv2 as cv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="image path")
ap.add_argument("-sp", "--shape-predictor", help="path to shape predictor file")
args = vars(ap.parse_args())

# load the image
img = cv.imread(args["image"])
img = cv.resize(img, (512,512), interpolation=cv.INTER_AREA)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# detector to get the bounding box of face
detector = dlib.get_frontal_face_detector()
# to get the facial landmarks
predictor = dlib.shape_predictor(args["shape_predictor"])


faces = detector(gray) # faces store the rectangular values of bounding box
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    cv.rectangle(img, (x1,y1), (x2, y2), (255, 0, 0), 3)
    cv.putText(img, "My Face with Facial Landmarks", (x1 - 20, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # landmark stores the objects of facial landmarks based on dlib library
    landmark = predictor(gray, face)
    for i in range(0, 67):
        x = landmark.part(i).x
        y = landmark.part(i).y

        cv.circle(img, (x, y), 3, (0, 0, 255), -1) # to diaplay the points

cv.imshow('image', img)

cv.waitKey(0)
cv.destroyAllWindows()