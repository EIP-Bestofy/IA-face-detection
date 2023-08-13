import face_recognition
import cv2 as cv


class FaceDetector:
    def detect_face(self, frame_small):
        face_locations = face_recognition.face_locations(frame_small)
        return face_locations

    def draw_square_at_locations(self, face_locations, frame):
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was scaled
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        return frame
