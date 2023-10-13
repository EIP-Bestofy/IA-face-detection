import cv2 as cv
import numpy as np
import tensorflow as tf

from PIL import Image
from keras.models import model_from_json
from src.detect_face import detect_face

#json_file = open('models/emotion_detection/emotion_model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#emotion_model = model_from_json(loaded_model_json)
#emotion_model.load_weights("models/emotion_detection/emotion_model.h5")
#emotion_dictionnary = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


emotion_model = tf.keras.models.load_model('models/emotion_detection/emotion_modelv2.h5')
emotion_dictionnary = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad']




def detect_emotion(frame):
    face_locations = detect_face(frame)

    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Crop the image where the head is located
        roi = frame[top:bottom, left:right]
        gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        # Resize to np.array of shape(48,48,1), this is the input shape of the model
        cropped_roi = np.expand_dims(np.expand_dims(cv.resize(gray_roi, (48, 48)), -1), 0)

        # Predict the emotion with my model
        #emotion_prediction = emotion_model.predict(cropped_roi)
        #emotion_index = int(np.argmax(emotion_prediction))

        # Predict the emotion with the model 
        predictions = emotion_model.predict(cropped_roi).argmax()
        state = emotion_dictionnary[predictions]

        if state == "Neutral":
            frame = frame[top:bottom, left:right]
            return frame
        else:
            display_info(state, [top, bottom, left, right], frame)
            return frame



        # Display the square location and the emotion
        #

def display_info(emotion, location, frame):
    # Draw Rectangle all around the head detected 
    cv.rectangle(frame, (location[2], location[0]), (location[3], location[1]), (0, 0, 255), 2)

    # Display the emotion detected
    cv.rectangle(frame, (location[2], location[0] - 20),
        (location[3], location[0]), (0, 0, 255), -1)
    cv.putText(frame, emotion, (location[2] + 3, location[0] - 5),
        cv.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 1,
        cv.LINE_AA)
