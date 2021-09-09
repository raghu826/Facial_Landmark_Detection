# Facial_Landmark_Detection
## Tools: `dlib`, `OpenCV`, `Python`

Detecting Facial Landmarks is a subset of shape prediction problem. The pre-trained [facial landmark detector](https://github.com/raghu826/Facial_Landmark_Detection/blob/main/shape_predictor_68_face_landmarks.dat) inside the dlib library is used to estimate the location of 68 points that map to facial structures on face.
The process onvolves two steps.
- First, localize the faces in an image using different techniques like Haar Cascades or dlib frontal face detector.
- Apply the shape predictor(facial landmark detector)  to obtain the coordinates of Region of Interest(ROI) in the face.

There are various applications of Facial Landmark detection namely
- Blink detection
- Head Pose estimation
- Face part extraction



