# Face Recongition with pictures / frames.

from PIL import Image , ImageDraw
import face_recognition

# Instructs the program to recognise a face by providing details of the picture with the imageSample variable.
# The variable and the index of 0 is used to represent 1 image is used for encoding.  
# .JPG is used because its a practical compressed image format for containing digital images.It's a compression ratio of 10:1 application.

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
    elonEncodings,
    jamesEncodings,
    michelleEncodings,
    emmaEncodings
]

# It indicates the algorithm what to call encoded faces in images.
memorisedNames = [
   "Elon Musk",
   "LeBron James",
   "Michelle Obama",
   "Emma Watson"
]

# Image that will be tested for the presence of a face.
imageSample = face_recognition.load_image_file("Images/Elonmusk3.jpg")
# Identifies the pixel locations of the face in the imageSample.
faceLocations = face_recognition.face_locations(imageSample)                 
faceEncodings = face_recognition.face_encodings(imageSample, faceLocations); pilImage = Image.fromarray(imageSample); draw = ImageDraw.Draw(pilImage)

# Gets each face from our sample image.
# Gets our coordinates for each face in our image.
for (top,right,bottom, left), face_encoding in zip(faceLocations, faceEncodings):
    # Makes the variable responsible for comparing the sample faces with the known faces.
    comparison = face_recognition.compare_faces(memorisedFaceEncodings,face_encoding)
    # face_recognition.compare_faces returms a list with a bunch of unique values relating to face recogntion match.


    # Used as a default string if the algorithm could not find a match.
    currentName = "Unknown"
    # Check if the identified faces matches with the defined faces.
    if True in comparison:
        firstMatchIndex = comparison.index(True)
        # Overwrite the name unkown with watever is in the first match index.
        currentName = memorisedNames[firstMatchIndex]

    # Draws rectangle.  
    draw.rectangle(((left, top), (right , bottom )) , outline= (150,150,150))

    # Creates text with function draw and name.
    textWidth,textHeight = draw.textsize(currentName)

    # Fill and outline are used to create colours of the rectangle.
    draw.rectangle(((left, bottom - textHeight - 10), (right , bottom)), fill=(0,0,255),outline=(0,0,255))
    # Defines the height and the width of the text.
    # Fill creates the colour.
    draw.text((left + 6 , bottom - textHeight - 5) , currentName, fill = (255,255,255,255))

# Releases the draw function for draw.rectangle and draw.text thus creating the rectangle and text.
del draw

pilImage.show()