from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import cv2
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import sys
from sklearn.cluster import KMeans
from keras.preprocessing import image as why

def colour(com):
    if(com>180):
        col = 'light pale'
    if(180>com>150):
        col = 'pale'
    if(150>com>120):
        col = 'tanned'
    if(120>com>80):
        col = 'Brown'
    if(80>com>60):
        col = 'dark brown'
    if(com<60):
        col = 'black'
    return col

def get_dominant_color(image, k=4, image_processing_size = None):

    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)

    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)

    label_counts = Counter(labels)

    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)
 
image = cv2.imread('alot.jpg')
image2 = cv2.imread('222.jpg')
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

classifier = Sequential()
classifier.add(Conv2D(32, (3,3),
                      input_shape = (64, 64, 3),
                      activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer ='adam',
                       loss ='binary_crossentropy',
                       metrics = ['accuracy'])
from keras. preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('C:\\Users\\Imperator\\Documents\\Workshop_University\\Machine_Vision\\New_folder\\pic\\train',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('C:\\Users\\Imperator\\Documents\\Workshop_University\\Machine_Vision\\New_folder\\pic\\test',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
classifier.fit_generator(training_set,
                         steps_per_epoch = 50,
                         epochs =2,
                         validation_data = test_set,
                         validation_steps = 10)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)
faces2= faceCascade.detectMultiScale(
    gray2,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

facei=0
a=0
store1 = [0]*31
prelist = ['0']*100
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    face = image[y:y +h , x:x + w]
    ganja = face

    img = face
    face = cv2.resize(image,(64,64))
    face= np.array(face).reshape((64,64,3))
    face = why.img_to_array(face)
    face = np.expand_dims(face, axis = 0)
    result = classifier.predict(face)
    facei = int(facei)
    facei = facei+1
    facei = str(facei)
    training_set.class_indices
    if result[0][0] >= 0.5:
        prediction ='female'
    else:
        prediction ='male'
    cv2.putText(image, facei , (x+w,y+h-20),cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255))
    a=a+1
    dominant = [0]*3
    dominant = get_dominant_color(ganja)
    facei = int(facei)
    prelist[facei]= prediction
    store1[facei] = dominant


    
facei = int(facei)
if(facei>1):
    print('more than one face detcted')
        

facei2=0
store2 = [0]*31
prelist2 = ['0']*100
for (x, y, w, h) in faces2:
    cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    face2 = image2[y:y +h , x:x + w]
    rgb2 = face2
    img = face2
    face2 = cv2.resize(image,(64,64))
    face2= np.array(face2).reshape((64,64,3))
    face2 = why.img_to_array(face2)
    face2 = np.expand_dims(face2, axis = 0)
    result2 = classifier.predict(face2)
    facei2 = int(facei2)
    facei2 = facei2+1
    facei2 = str(facei2)
    training_set.class_indices
    if result2[0][0] >= 0.5:
        prediction ='female'
    else:
        prediction ='male'
    cv2.putText(image2, facei2 , (x+w,y+h-20),cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255))
    a=a+1
    dominant2 = [0]*3
    dominant2 = get_dominant_color(rgb2)
    facei2 = int(facei2)
    prelist2[facei2]= prediction
    store2[facei2] = dominant2

    

facei2 = int(facei2)
if(facei2>1):
    print('more than one face detected')

facet = facei2+facei

cv2.imshow('first image',image)
cv2.imshow('second image',image2)
cv2.waitKey(1)
opindex1 = 1
opindex2 = 1
repeat = 0
repeatin = 'n'
n=999

for repeat in range(n):
    
    if(facei>1):
        print('Please look at "first image" and decide')
        print('if you want to compare skin color with other person in the photo input 0')
        print('if you want to compare skin color with the second photo, pick the person number you want to compare(the number is beside their face)')
        opindex1 = input()
        opindex1 = int(opindex1)
        print(opindex1)

    if(facei2>1):
        print('Please look at "second image image" and decide')
        print('if you want to compare skin color with other person in the photo input 0')
        print('if you want to compare skin color with the first photo, pick the person number you want to compare(the number is beside their face)')
        opindex2 = input()
        opindex2 = int(opindex2)
        print(opindex2)
    fopindex1=0
    fopindex2=0

    if(opindex1 == 0):
        print('pick the first person number you want to compare(the number is beside their face)')
        fopindex1= input()
        fopindex1 = int(fopindex1)
        print('pick the second person number you want to compare(the number is beside their face)')
        fopindex2 = input()
        fopindex2 = int(fopindex2)
        com = np.mean(store1[fopindex1])
        com2 = np.mean(store1[fopindex2])
        if(com>com2):
            print('for the first pic the person with', fopindex1, 'index is whiter')
        else:
            print(fopindex2)
            print('for the first pic the person with',fopindex2,' index is whiter')
    
        if(prelist[fopindex1]=='female'):
            com = com-10
        if(prelist[fopindex2]=='female'):
            com2 = com2-10
        col= colour(com)
        col2 = colour(com2)
        print('for first photo')
        print('The first person is',col, ' the gender is ', prelist[fopindex1])
        print('The second person is',col2, ' the gender is ', prelist[fopindex2])

    if(opindex2 == 0):
        print('pick the first person number you want to compare(the number is beside their face)')
        fopindex1= input()
        fopindex1 = int(fopindex1)
        print('pick the second person number you want to compare(the number is beside their face)')
        fopindex2 = input()
        fopindex2 = int(fopindex2)
        com = np.mean(store2[fopindex1])
        com2 = np.mean(store2[fopindex2])
        if(com>com2):
            print('for the second pic the person with', fopindex1, 'index is whiter')
        else:
            print(fopindex2)
            print('for the second pic the person with',fopindex2,' index is whiter')
    
        if(prelist2[fopindex1]=='female'):
            com = com-5
        if(prelist2[fopindex2]=='female'):
            com2 = com2-5
        col= colour(com)
        col2 = colour(com2)
        print('for second photo')
        print('The first person is',col,' the gender is ', prelist2[fopindex1])
        print('The second person is',col2,' the gender is ', prelist2[fopindex1])

    if(opindex1 !=0):
        if(opindex2 != 0):
            com = np.mean(store1[opindex1])
            com2 = np.mean(store2[opindex2])
            if(com>com2):
                print('the person from the first pic with', opindex1, 'index is whiter')
            else:
                print(fopindex2)
                print('the person from the second pic with',opindex2,' index is whiter')
    
            if(prelist[opindex1]=='female'):
                com = com-10
            if(prelist2[opindex2]=='female'):
                com2 = com2-10
            col= colour(com)
            col2 = colour(com2)
            print('The first person from first pic is',col,' the gender is ', prelist[opindex1])
            print('The second person from second pic is',col2,' the gender is ',prelist2[opindex2])
        else:
            print()
            print('sorry, you are comparing image one with image one, yet ask to compare image one to image two')
            print('please do the comparison one by one')
        
    if(opindex1 == 0):
        if(opindex2 !=0):
            print()
            print('sorry, you are comparing image one with image one, yet ask to compare image one to image two')
            print('please do the comparison one by one')
    if(facet<3):
        print('note that woman on average are whiter than man, so their white standard is a bit higher')
        print('')
        print('')
        break
    if(facet>=3):
        print('note that woman on average are whiter than man, so their white standard is a bit higher')
        print('do you want to continue comparing image(y/n)')
        repeatin = input()
        print('')
        print('')
        print('')
        print('')
        if(repeatin =='n'):
            break


    

        
        
        
