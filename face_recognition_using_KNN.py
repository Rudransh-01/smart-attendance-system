# recognise faces using some classification algorithm - like logistics, SVM, KNN etc.

# read a video stream
# extract the faces out of it
# load the training data (numpy array of all the persons)
		# x- values will be stored in the numpy arrays
		# y- values we will have to assign for each person
# use KNN to find the prediction of the face (int)
# map the predicted id to the name of the user
# name the predictions on the screen - bounding box and name

import cv2
import numpy as np 
import os

########## KNN CODE #####################################
def distance(v1, v2):
	#Eucldian
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []

	for i in range(train.shape[0]):
		#get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]

		#compute the distance from test point
		d = distance(test, ix)
		dist.append([d,iy])

		#sort based on distance and get top k
		dk = sorted(dist, key = lambda x:x[0])[:k]

		#retrieve only the labels
		labels = np.array(dk)[:,-1]

		#get frequencies of each label
		output = np.unique(labels, return_counts = True)

		#find maximum frequency and corresponding labels
		index = np.argmax(output[1])
		return output[0][index]

###########################################################


skip = 0
dataset_path = 'data/'

# x of the data
face_data = []
# y of the data
labels = []
# labels for the given file
class_id = 0
# to create mapping between the id - name
names = {}

# data preparation
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		print("loading..."+fx)
		names[class_id] = fx[:-4]
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

	#create labels for the class
	target = class_id*np.ones((data_item.shape[0],))
	class_id += 1
	labels.append(target)


face_dataset = np.concatenate(face_data, axis = 0)
face_labels = np.concatenate(labels, axis = 0).reshape((-1,1))
print(face_dataset.shape, face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels), axis = 1)
print(trainset.shape)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

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
        #extract - crop out the required part also called as region of interest
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        out = knn(trainset, face_section.flatten())

        #display on screen name and rectangle around it
        predicted_name = names[int(out)]
        cv2.putText(gray_frame,predicted_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,0,0),2)


    cv2.imshow("Face Recognition", gray_frame)

	#wait for user to press a key -q, on which you will break the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

