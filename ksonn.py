#####

# KSONN model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from minisom import MiniSom
  
  # Data Preprocessing
df1=pd.read_csv("/home/divum/all_dataframe.csv")
df1.head()
print(len(df1))
plt.figure(figsize=(8, 8))
plt.plot(df1, 'bo')
plt.show()
df2=pd.read_csv("/home/divum/Downloads/bandgap.csv")
df2.head()
print(len(df2))
plt.figure(figsize=(8, 8))
plt.plot(df2, 'ro')
plt.show()

  ###Feature Scaling
print(df1.describe())
print(df2.describe())

  ##Training Data
x_train,x_test,y_train,y_test=train_test_split(df1,df2,test_size=0.2,random_state=None)

  ##Preparing Model
sc=MinMaxScaler(feature_range=(0,94))
x=sc.fit_transform(x_train,y_train)
print(x)

  ##Hyperparameters
# x, y = 10, 10 dimensions---> grid
#input_len=no of attributes (93)
# sigma = 1.
# learning_rate = 0.5
# epochs = 50000
# decay_parameter = epochs / 2

som=MiniSom(x=10,y=10,input_len=93,sigma=1.0,learning_rate=0.5)

  ##Initialize weights and train
som.random_weights_init(x)
som.train_random(data=x,num_iteration=100)

  ##mapping winner weight
mappings=som.win_map(x)
# print(mappings)
bandgap=np.concatenate((mappings[(7,8)],mappings[(3,1)],mappings[(5,1)]),axis=0)
print(bandgap)
bandgap=sc.inverse_transform(bandgap)
print('########################################')
print("predicted bandgap=",bandgap)