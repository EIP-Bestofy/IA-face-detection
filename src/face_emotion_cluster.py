import numpy as np

class Cluster: 
    def __init__(self):
        self.info_array = []

    def face_emotion_cluster(self, emotions, locations):
        for emotion, location in zip(emotions, locations):
            y_location = (location[0] + location[2]) / 2
            x_location = (location[1] + location[3]) / 2
            if not self.info_array:
                self.info_array.append([[x_location, y_location, emotion]])

            idx = self.choose_cluster(x_location, y_location)
            if idx == -1:
                self.info_array.append([[x_location, y_location, emotion]])
            else:
                self.info_array[idx].append([x_location, y_location, emotion])


        
    def choose_cluster(self, x_location, y_location):
        for idx, cluster in enumerate(self.info_array):
            mean_x = sum([c[0] for c in cluster]) / len(cluster)
            mean_y = sum([c[1] for c in cluster]) / len(cluster)
            # In the mean, return the index
            if ((x_location < (mean_x + 50) and x_location > (mean_x - 50))
                and y_location < (mean_y + 50) and y_location > (mean_y - 50)):
                return idx
        return -1

    def get_info_array(self):
        return self.info_array

