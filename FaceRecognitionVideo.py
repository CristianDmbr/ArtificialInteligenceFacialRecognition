# Facial Recognition in live recording.

import face_recognition
import numpy as np
import cv2

video_Capture  = cv2.VideoCapture(0)

# Instructs the program to recognise a face by providing details of the picture with the Image variable.
# The variable and the index of 0 can be used to represent 1 image, 
 
# Me               
cristianImage = face_recognition.load_image_file("Images/Cristian1.jpg")
cristianEncodings = face_recognition.face_encodings(cristianImage) [0] 

# Caucasian man Example                   
elonSample = face_recognition.load_image_file("Images/Elon.jpg")
elonEncodings = face_recognition.face_encodings(elonSample) [0]

# Black man Example
jamesSample = face_recognition.load_image_file("Images/LebronJames.jpg")
jamesEncodings = face_recognition.face_encodings(jamesSample) [0]

# Black woman Example
michelleSample = face_recognition.load_image_file("Images/MichelleObama.jpg")
michelleEncodings = face_recognition.face_encodings(michelleSample) [0]

# Caucasian woman Example
emmaSample = face_recognition.load_image_file("Images/EmmaWatson1.jpg")
emmaEncodings = face_recognition.face_encodings(emmaSample) [0]

memorisedFaceEncodings = [                            
    cristianEncodings,
    elonEncodings,
    jamesEncodings,
    michelleEncodings,
    emmaEncodings,

]

# It indicates the software what to call the memorised faces in the video.
learnedNames = [
   "Cristian",
   "Elon Musk",
   "Lebron james",
   "Michelle Obama",
   "Emma Watson",
]

while True:
    ret , frame = video_Capture.read()

    rgbFrame = frame[:, :, ::-1]

    faceLocations = face_recognition.face_locations(rgbFrame)
    faceEncodings = face_recognition.face_encodings(rgbFrame, faceLocations)

    for (top, right, bottom, left), faceEncodings in zip(faceLocations, faceEncodings):
        # Makes the variable responsible for comparing the sample faces with the known faces.
        matches = face_recognition.compare_faces(memorisedFaceEncodings,faceEncodings)
        # face_recognition.compare_faces returms a list with a bunch of useful numbers relating to face recogntion matching.


        # Used as a default value if the algorithm could not find a match
        currentName = "Unknown"
        # Check if the identified faces matches with the defined faces
        if True in matches:
           first_match_index = matches.index(True)
           # Overwrite the name unkown with watever is in the first match index
           currentName = learnedNames[first_match_index]

           cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)
           # Selected font
           font = cv2.FONT_HERSHEY_DUPLEX
           cv2.putText(frame,currentName, ( left + 6, bottom - 6), font, 1.0, (255,255,255), 1)
    

    cv2.imshow("Video", frame)

    # Closes the window with pressing q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_Capture.release()
cv2.destroyAllWindows()