
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#data augmentation for the training variable
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

x_train=train_datagen.flow_from_directory('/content/dataset/Training',
                                          target_size = (64,64),
                                          class_mode = 'categorical',
                                          batch_size = 100)

x_test=test_datagen.flow_from_directory('/content/dataset/Testing',
                                          target_size = (64,64),
                                          class_mode = 'categorical',
                                          batch_size = 100)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Convolution2D,MaxPooling2D,Flatten

#adding layers
model = Sequential()
model.add(Convolution2D(32,(3,3),activation = 'relu',input_shape = (64,64,3))) #convultion layer
model.add(MaxPooling2D(pool_size = (2,2))) #maxpooling layer
model.add(Flatten()) #flatten layer

model.add(Dense(300,activation = 'relu')) #hidden layer1
model.add(Dense(150,activation = 'relu')) #hidden layer2
model.add(Dense(4,activation = 'softmax')) #output layer

#compile the model
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

#training the model
model.fit_generator(x_train,steps_per_epoch = len(x_train),epochs = 10,validation_data = x_test,validation_steps = len(x_test))

#save the model
model.save('animal.h5')

from tensorflow.keras.preprocessing import image
import numpy as np

img= image.load_img('/content/dataset/Testing/elephants/Z (1).jpeg',target_size=(64,64))

img

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred=np.argmax(model.predict(x))
op=['bears','crows','elephants','rats']
op[pred]

img=image.load_img('/content/dataset/Testing/bears/m9.jpeg',target_size=(64,64))

img

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred1=np.argmax(model.predict(x))
op=['bears','crows','elephants','rats']
op[pred1]

img=image.load_img('/content/dataset/Testing/crows/Z1  (30).jpg',target_size=(64,64))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred2=np.argmax(model.predict(x))
op=['bears','crows','elephants','rats']
op[pred2]

img=image.load_img('/content/dataset/Testing/rats/images (100).jpeg',target_size=(64,64))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred3=np.argmax(model.predict(x))
op=['bears','crows','elephants','rats']
op[pred3]