#Library Importing
import numpy as np
import pandas as pd
import keras
import tensorflow
import sklearn
#Data Importing
data=pd.read_excel("Concrete_Data.xls")
X=data.drop(['Concrete compressive strength(MPa, megapascals) '],axis=1)
y=data['Concrete compressive strength(MPa, megapascals) ']


#Data Preprocessing

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Model Initialization
from keras.layers import Dense
from keras.models import Sequential
Neural_Network=Sequential()
Neural_Network.add(Dense(units=4,activation='relu',kernel_initializer='uniform',input_dim=8))
Neural_Network.add(Dense(units=4,activation='relu',kernel_initializer='uniform'))
Neural_Network.add(Dense(units=1,activation='relu',kernel_initializer='uniform'))
Neural_Network.compile(optimizer='adam',loss='mean_squared_error')

## Model Fitting and Prediction

Neural_Network.fit(X_train,y_train,batch_size=32, epochs=100)
pred=Neural_Network.predict(X_test)
from sklearn.metrics import mean_squared_error
print("\n Mean Squared Error:",mean_squared_error(y_test,pred))

## Model Saving

from keras.models import model_from_json

# serialize model to JSON
model_json = Neural_Network.to_json()
with open("Neural_Network.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
Neural_Network.save_weights("Neural_Network.h5")
print("Saved model to disk")



