import face_recognition
import cv2 as cv
from PIL import Image, ImageDraw
import numpy as np

class LandmarksDetector:
    def detect_landmarks(self, frame_small):
        face_landmarks_list = face_recognition.face_landmarks(frame_small)
        return face_landmarks_list

    def draw_landmarks(self, face_landmarks_list, frame, details):
        # Create a PIL imagedraw object so we can draw on the picture
        pil_frame = Image.fromarray(frame)
        d = ImageDraw.Draw(pil_frame)

        for face_landmarks in face_landmarks_list:
            if details:
                # Make the eyebrows into a nightmare
                d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
                d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
                d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
                d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

                # Gloss the lips
                d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
                d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
                d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
                d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

                # Sparkle the eyes
                d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
                d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

                # Apply some eyeliner
                d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
                d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
            else:
                # Let's trace out each facial feature in the image with a line!
                for facial_feature in face_landmarks.keys():
                    d.line(face_landmarks[facial_feature], width=5)

        frame = np.array(pil_frame)

        return frame
