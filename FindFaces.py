import os
import face_recognition
image = face_recognition.load_image_file(r"C:\Users\ugur_\Desktop\Introduction to Biometrics\Assigmment\FaceRecognitionAssiggnment\FaceRecognition\Images\Data\Team1.jpeg")
face_locations = face_recognition.face_locations(image) #Array of coords of each face

print(f'There are {len(face_locations)} people in this frame')






