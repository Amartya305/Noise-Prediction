import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
#preprocessing
sensorData = pd.read_csv('..\data\Sensor_record_20200118_005002_AndroSensor.csv',sep=';',skiprows = 1 ,header=0)
target = 'SOUND LEVEL (dB)'
features = ['LOCATION Latitude : ','LOCATION Longitude : ']
Y = sensorData[target]
X = sensorData[features]

#training
regressor = LinearRegression()
regressor.fit(X,Y)
print(regressor)
pickle.dump(regressor,open('..\models\multiple_regressor.pkl','wb'))
