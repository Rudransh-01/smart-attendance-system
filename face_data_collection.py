import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = 'data/'
file_name = input("Enter the name of person : ")
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue



    #scaling factor and no of neighbours
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    #above will return tuple with face coordinates that we will have to draw 
    faces = sorted(faces,key = lambda f:f[2]*f[3])

    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,0,0),2)
        #extract - crop out the required part also called as region of interest
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))


        skip += 1

        if ( skip%10 == 0 ):
            face_data.append(face_section)
            print(len(face_data))
            #store every 10th face



        #cv2.imshow("BGR Frame",frame)
        cv2.imshow("GRAY Frame", gray_frame)
        cv2.imshow("Face section", face_section)

    #wait for user to press a key -q, on which you will break the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# save the data
np.save(dataset_path+file_name+'.npy',face_data)
print("Data successfully saved at " + dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()