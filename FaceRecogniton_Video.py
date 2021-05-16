import face_recognition
import os
import cv2
from numba import jit, cuda

KNOWN_FACES_DIR="Images/LCDP"
TOLERANCE=0.5
MODEL="hog"
Frame_Thickness=1
Font_Thickness=2


input_movie = cv2.VideoCapture("Images\Video\Example.avi")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

video=cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('Example.avi', video, 29.97, (2560, 1440))

lmm_image = face_recognition.load_image_file(r"Images\LCDP\Photos\Nairobi.jpg")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file(r"Images\LCDP\Photos\Tokyo_1.jpg")
al_face_encoding = face_recognition.face_encodings(al_image)[0]

Berlin_image = face_recognition.load_image_file(r"Images\LCDP\Photos\Berlin.jpg")
Berlin_face_encoding = face_recognition.face_encodings(Berlin_image)[0]

Denver_image = face_recognition.load_image_file(r"Images\LCDP\Photos\Denver.jpg")
Denver_face_encoding = face_recognition.face_encodings(Denver_image)[0]

# Helsinki_image = face_recognition.load_image_file(r"Images\LCDP\Photos\Helsinki.jpg")
# Helsinki_face_encoding = face_recognition.face_encodings(Helsinki_image)[0]

Moscow_image = face_recognition.load_image_file(r"Images\LCDP\Photos\Moscow.jpg")
Moscow_face_encoding = face_recognition.face_encodings(Moscow_image)[0]

Oslo_image = face_recognition.load_image_file(r"Images\LCDP\Photos\Oslo.jpg")
Oslo_face_encoding = face_recognition.face_encodings(Oslo_image)[0]

Proffesour_image = face_recognition.load_image_file(r"Images\LCDP\Photos\Proffesour.jpg")
Professour_face_encoding = face_recognition.face_encodings(Proffesour_image)[0]

Stockholm_image = face_recognition.load_image_file(r"Images\LCDP\Photos\Stockholm.jpg")
Stockholm_face_encoding = face_recognition.face_encodings(Stockholm_image)[0]

known_faces=[
    lmm_face_encoding,
    al_face_encoding,
    Berlin_face_encoding,
    Denver_face_encoding,
    # Helsinki_face_encoding,
    Moscow_face_encoding,
    Oslo_face_encoding,
    Professour_face_encoding,
    Stockholm_face_encoding

]
known_names=[
    'Nairobi',
    "Tokyo",
    "Berlin",
    "Denver",
    # "Helsinki",
    "Moscow",
    "Oslo",
    "Professour",
    "Stockholm"
]

frame_number = 0

while True:
    
    ret, frame=input_movie.read()
    frame_number += 1

    if not ret:
        break


    locations=face_recognition.face_locations(frame, model=MODEL)
    encodings=face_recognition.face_encodings(frame, locations)
     
    for face_encoding, face_location in zip(encodings, locations):
        results=face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match="unknown person"
       
        if True in results:
            match=known_names[results.index(True)]
            
            top_left=(face_location[3], face_location[0])
            bottom_right=(face_location[1], face_location[2])
            color=[0,255,0]
            cv2.rectangle(frame,top_left, bottom_right, color, Frame_Thickness)

            top_left=(face_location[3], face_location[2])
            bottom_right=(face_location[1],face_location[2]+22)
            cv2.rectangle(frame, top_left, bottom_right, color)
            cv2.putText(frame,match,(face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200), Font_Thickness)
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)
    cv2.imshow("Result",frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        input_movie.release()
        cv2.destroyAllWindows()
        break
    
            










