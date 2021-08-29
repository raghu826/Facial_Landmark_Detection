import dlib
#import imutils
import cv2 as cv
import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help="image path")
# args = vars(ap.parse_args())

cap = cv.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv.rectangle(frame, (x1,y1), (x2, y2), (255, 0, 0), 3)
        cv.putText(frame, "My Face with Facial Landmarks", (x1 - 20, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        landmark = predictor(gray, face)
        for i in range(0, 67):
            x = landmark.part(i).x
            y = landmark.part(i).y

            cv.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv.imshow('video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()








