import cv2 as cv

class Display_Info:

    def display(self, emotions, locations, frame):
        for emotion, location in zip(emotions, locations):
            # Draw Rectangle all around the head detected 
            frame = cv.rectangle(frame, (location[2], location[0]), (location[3], location[1]), (0, 0, 255), 2)
            
            # Display the emotion detected
            frame = cv.rectangle(frame, (location[2], location[0] - 20),
                (location[3], location[0]), (0, 0, 255), -1)
            frame = cv.putText(frame, emotion, (location[2] + 3, location[0] - 5),
                cv.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1,
                cv.LINE_AA)
        return frame
