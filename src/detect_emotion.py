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
    locations = []
    emotions_states = []
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

        # Predict the emotion with the model 
        predictions = emotion_model.predict(cropped_roi, verbose=0).argmax()
        state = emotion_dictionnary[predictions]

        #if state == "Neutral":
        #    frame = frame[top:bottom, left:right]
        #    return frame
        locations.append([top, bottom, left, right])
        emotions_states.append(state)
    
    locations= np.array(locations)
    emotions_states= np.array(emotions_states)
    return emotions_states, locations
