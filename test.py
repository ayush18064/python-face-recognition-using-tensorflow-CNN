import tensorflow as tf
from tf_keras.models import load_model
import numpy as np
import cv2

# Load the pre-trained model
model = load_model("keras_model.h5")

# Initialize face detector and video capture
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX


# Function to get class name
def get_class_name(class_index):
    if class_index == 0:
        return "ayush"
    elif class_index == 1:
        return "lokesh"
    else:
        return "unknown"


# Confidence threshold
confidence_threshold = 0.7

while True:
    success, img_original = cap.read()
    if not success:
        break

    # Detect faces in the image
    faces = facedetect.detectMultiScale(img_original, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop and preprocess the face region
        crop_img = img_original[y:y + h, x:x + w]
        img_resized = cv2.resize(crop_img, (224, 224))
        img_normalized = img_resized.astype('float32') / 255.0  # Normalize the image
        img_reshaped = np.reshape(img_normalized, (1, 224, 224, 3))

        # Predict the class and confidence
        prediction = model.predict(img_reshaped)
        class_index = np.argmax(prediction, axis=1)[0]
        probability_value = np.max(prediction)

        # Check confidence and display result
        if probability_value > confidence_threshold:
            class_name = get_class_name(class_index)
            cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img_original, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(img_original, class_name, (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_original, f"{probability_value * 100:.2f}%", (x, y - 30), font, 0.75, (255, 0, 0), 2,
                        cv2.LINE_AA)

    # Show the result
    cv2.imshow("Result", img_original)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
