import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import randrange


# Load the cascade
face_cascade = cv2.CascadeClassifier(r'C:\Users\ugur_\Desktop\haarcascade_frontalface_default.xml')

#choose an image to detect faces in


img1=cv2.imread(r"C:\Users\ugur_\Desktop\IMG5.jpeg")
grayscaled_img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

face_coordinates=face_cascade.detectMultiScale(grayscaled_img1)

print(face_coordinates)

count=0

for(x,y,w,h) in face_coordinates:
    cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),5)
    count+=1
    
cv2.imshow("Result",img1)
cv2.waitKey()
cv2.destroyAllWindows()

print("There are "+str(count)+" person on the image")

 
cap = cv2.VideoCapture(r'C:\Users\ugur_\Desktop\Video.mp4')
countPerson=0
countFrame=0
frame_width=int(cap.get(3))
frame_height=int(cap.get(4))

out=cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
while True:
    countFrame+=1
    # Read the frame
    successful_frame_read, frame=cap.read()
    # Convert to grayscale
    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Coverts image to grayscale
    # Detect the faces
    face_coordinates=face_cascade.detectMultiScale(grayscaled_img)
    # Draw the rectangle around each face
    
    
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count+=1
        
    # Display
    
    
    cv2.imshow('img', frame)
    out.write(frame)
    
    
    
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#• Release the VideoCapture object
cap.release()
out.release()
cv2.destroyAllWindows()
countresult=countPerson/countFrame

print(countresult)




print("code completed")