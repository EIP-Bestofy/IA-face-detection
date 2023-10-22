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
        self.count_detection_face = 0
        self.count_detection_attempt = 0
        
    def detect(self, frame):
        locations = []
        emotions_states = []
        self.count_detection_attempt += 1

        face_locations = self.detect_face.detect(frame)
        for top, right, bottom, left in face_locations:
            self.count_detection_face += 1
            
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
        print(self.count_detection_face)
        print(self.count_detection_attempt)
        clusters = self.cluster.get_info_array()
        print("Start Display Cluster INFO")
        for idx, cluster in enumerate(clusters):
            np_cluster = np.array(cluster)
            np_cluster_pos = np_cluster[:, :2].astype(float).astype(int)
            np_cluster_emo = np.array(np_cluster[:, 2])

            # percentage detction between every cluster
            percentage_detectin_face = np.around((np_cluster.shape[0] * 100) / self.count_detection_face, 1)
            # percentage detection over every detection attempt
            percentage_detection_attempt = np.around((np_cluster.shape[0] * 100) / self.count_detection_attempt, 1)

            # Mean of the position of the first cluster
            means = np.mean(np_cluster_pos[:, :2], axis=0)
            x_mean = int(round(means[0]))
            y_mean = int(round(means[1]))
            

            # Pourcentage of each emotions detected
            unique_emos, counts = np.unique(np_cluster_emo, return_counts=True)
            percentages = np.around(counts / np_cluster_emo.size * 100, 1)
            emotion_percentages = dict(zip(unique_emos, percentages))

            print("Cluster", idx, ":")
            print("\tPercentage detection over every attempt", percentage_detection_attempt)
            print("\tPercentage detection over every face", percentage_detectin_face)
            print("\tMean X", x_mean, "Mean Y", y_mean)
            print("\tPercentage emotions detected", emotion_percentages)


    def add_info_cluster(self, emotions_states, locations):
        self.cluster.face_emotion_cluster(emotions_states, locations)