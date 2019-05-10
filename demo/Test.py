import scipy.io as sio
import numpy as np
from keras.optimizers import SGD,Adam,RMSprop
from keras.layers.core import Dense, Activation
from keras.models import Sequential,save_model,load_model
from sklearn import preprocessing
from keras.utils import to_categorical


X = np.load('X_markergenes_profile_D1.npy')     # X_markerspecies_profile_D1; X_markergenes_profile_D1
Y = np.load('lables_D1.npy')
X_test = np.load('X_markergenes_profile_D2.npy')  # X_markerspecies_profile_D2;  X_markergenes_profile_D2
Y_test = np.load('lables_D2.npy')


X = np.array(X)
Y = np.array(Y)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

num_classes = 2
'''
batch_size = 2 # 2,3,4
epochs = 300   # 70,80,90 100, 110, 120,..., 200,...250,300,350
(nsize,nf) = X.shape
'''

Y_train = to_categorical(Y,num_classes)
Y_test = to_categorical(Y_test,num_classes)

X_train= (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))       # X_train = (X - X.mean(axis=0)) / X.std(axis=0)
X_test = (X_test - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # X_test = (X_test - X.mean(axis=0)) / X.std(axis=0)

'''
model = Sequential()
model.add(Dense(input_dim=nf, units=18))
model.add(Activation('relu'))
model.add(Dense(9))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))                   # relu

model.compile(loss='categorical_crossentropy',
               optimizer=RMSprop(),
               metrics=['accuracy'])    # categorical_crossentropy  mse

'''
#model.fit(X_train, Y_train,batch_size=batch_size,epochs=epochs,verbose=1)

model = load_model('MarkerGenes_D1_model.h5')  # MarkerSpecies_D1_model.h5;  MarkerGenes_D1_model.h5

scores = model.evaluate(X_test, Y_test)
print(scores)