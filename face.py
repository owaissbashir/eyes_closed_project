import cv2
import numpy as np
import cv2
import dlib
import numpy as np
from imutils import face_utils

# Load the pre-trained eye cascade classifier
eyes_open='images/eyes_open.jpg'
eyes_closed='images/eyes_closed.jpg'
def face_swap(eyes_open,eyes_closed):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    image = cv2.imread(eyes_open)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform eye detection
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Create a canvas to display both eyes
    canvas = image.copy()

    # Slice out the left and right eyes
    if len(eyes) >= 2:
        (x, y, w, h) = eyes[0]
        eyes_roi_left = image[y:y+h, x:x+w]

        (x, y, w, h) = eyes[1]
        eyes_roi_right = image[y:y+h, x:x+w]

        # Resize the eye images to have the same height
        max_height = max(eyes_roi_left.shape[0], eyes_roi_right.shape[0])
        eyes_roi_left = cv2.resize(eyes_roi_left, (int(eyes_roi_left.shape[1] * max_height / eyes_roi_left.shape[0]), max_height))
        eyes_roi_right = cv2.resize(eyes_roi_right, (int(eyes_roi_right.shape[1] * max_height / eyes_roi_right.shape[0]), max_height))

        # Concatenate the eye images horizontally
        eyes_combined = np.hstack((eyes_roi_right, eyes_roi_left))

    #     # Draw rectangles around the eyes
    #     for (x, y, w, h) in eyes:
    #         cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        # cv2.imshow('Eyes Combined', eyes_combined)
    else:
        print("Both eyes not detected.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        pass
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
    # cv2.imshow('image',image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    face_eyes_opened = image[y:y+h, x:x+w]
    # cv2.imshow('face_eyes_show',face_eyes_opened)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Load the facial landmark predictor
    predictor_path = 'shape_predictor_68_face_landmarks (2).dat'
    predictor = dlib.shape_predictor(predictor_path)

    # Load the image
    image = cv2.imread(eyes_closed)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a face detector
    detector = dlib.get_frontal_face_detector()

    # Perform face detection
    rectangles = detector(gray, 0)

    # Iterate over the detected faces
    for rectangle in rectangles:
        # Extract the face region
        (x, y, w, h) = (rectangle.left(), rectangle.top(), rectangle.width(), rectangle.height())
        face = image[y:y+h, x:x+w]
        
        # Resize the face_eyes_opened image to match the size of the face region
        resized_face_eyes_opened = cv2.resize(face_eyes_opened, (w, h))

        # Draw rectangle around the detected face
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Replace the face region with the resized face_eyes_opened image
        image[y:y+h, x:x+w] = resized_face_eyes_opened



    return image

# Display the image with the detected face


# image=face_swap(eyes_open,eyes_closed)
# cv2.imshow('Detected Face', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


