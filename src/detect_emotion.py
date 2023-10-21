import cv2 as cv
import numpy as np
import tensorflow as tf
from src.detect_face import Detect_face
from src.display_info import Display_Info
from src.face_emotion_cluster import Cluster


class Detect_Emotion():
    def __init__(self):
        self.model = tf.keras.models.load_model('models/emotion_detection/emotion_modelv2.h5')
        self.dictionnary = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad']
        self.display_emotion = Display_Info()
        self.detect_face = Detect_face()
        self.cluster = Cluster()
        
    def detect(self, frame):
        face_locations = self.detect_face.detect(frame)
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
            predictions = self.model.predict(cropped_roi, verbose=0).argmax()
            state = self.dictionnary[predictions]

            #if state == "Neutral":
            #    frame = frame[top:bottom, left:right]
            #    return frame
            locations.append([top, bottom, left, right])
            emotions_states.append(state)

        return emotions_states, locations
    
    def display_info(self, emotions, locations, frame):
        return self.display_emotion.display(emotions, locations, frame)

    def print_cluster_info(self):
        clusters = self.cluster.get_info_array()
        print("Start Display Cluster INFO")
        for idx, cluster in enumerate(clusters):
            np_cluster = np.array(cluster)
            np_cluster_pos = np_cluster[:, :2].astype(float).astype(int)
            np_cluster_emo = np.array(np_cluster[:, 2])

            means = np.mean(np_cluster_pos[:, :2], axis=0)
            x_mean = int(round(means[0]))
            y_mean = int(round(means[1]))
            print("Cluster", idx, ": Mean X", x_mean, "Mean Y", y_mean)

            unique_emos, counts = np.unique(np_cluster_emo, return_counts=True)
            percentages = np.around(counts / np_cluster_emo.size * 100, 1)
            emotion_percentages = dict(zip(unique_emos, percentages))
            print("\t", emotion_percentages)


    def add_info_cluster(self, emotions_states, locations):
        self.cluster.face_emotion_cluster(emotions_states, locations)