
# coding: utf-8

# In[49]:

from __future__ import print_function
import keras
from keras.datasets import mnist


from keras import backend as K
from keras.layers import Activation
import time
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt



from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1


# In[2]:

batch_size = 128
num_classes = 10
epochs = 8


# In[3]:

img_rows, img_cols = 30, 30


# In[4]:

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[29]:

classes = np.unique(y_train)
nClasses = len(classes)

print(classes)
print(nClasses)


# In[31]:

get_ipython().magic('matplotlib inline')


# In[32]:

plt.figure(figsize=[5,5])


# In[33]:

# Display the first image in training data
plt.subplot(121)
plt.imshow(x_train[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(y_train[0]))


# In[34]:

#converting 28x28 images to 28x28x1
x_train= x_train.reshape(-1, 28,28, 1)
x_test = x_test.reshape(-1, 28,28, 1)
x_train.shape, x_test.shape


# In[36]:

#Rescaling to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.


# In[37]:

#converting the class labels into a one-hot encoding vector.

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


# In[38]:

print('Original label:', y_train[0])
print('After conversion to one-hot:', y_train_one_hot[0])


# In[40]:

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x_train, y_train_one_hot, test_size=0.2, random_state=13)


# In[41]:

x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[ ]:

batch_size = 64
epochs = 30
num_classes = 10


# In[44]:

number_model = Sequential()
number_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
number_model.add(LeakyReLU(alpha=0.1))
number_model.add(MaxPooling2D((2, 2),padding='same'))
number_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
number_model.add(LeakyReLU(alpha=0.1))
number_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
number_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
number_model.add(LeakyReLU(alpha=0.1))                  
number_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
number_model.add(Flatten())
number_model.add(Dense(128, activation='linear'))
number_model.add(LeakyReLU(alpha=0.1))                  
number_model.add(Dense(num_classes, activation='softmax'))


# In[45]:

#compile model 
number_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[46]:

number_model.summary()


# In[52]:

number_train = number_model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))


# In[55]:

#testing the model
test_eval = number_model.evaluate(x_test, y_test_one_hot, verbose=0)


# In[62]:

np.reshape(y_test_one_hot,newshape=(10000,3))


# In[60]:

x_test.shape


# In[56]:

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


# In[65]:

accuracy = number_train.history['acc']
val_accuracy =number_train.history['val_acc']
loss = number_train.history['loss']
val_loss = number_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[64]:



