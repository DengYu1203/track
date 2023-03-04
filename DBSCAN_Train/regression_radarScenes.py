import os
import csv
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import pickle
import m2cgen as m2c

code_path = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(code_path,'radarScenes')

def train_model(Y,X,test_Y,test_X):
  print('Training Net!')
  x_ = X
  test_x_ = test_X
  model = MLPRegressor(hidden_layer_sizes=300, max_iter=300, activation='tanh')
  model.fit(x_,Y)
  
  model_path = os.path.join(code_path,'MLP_model.sav')
  pickle.dump(model,open(model_path,'wb'))
  print('Train: ',model.score(x_,Y))
  print('Test: ',model.score(test_x_,test_Y))
  predict_y = model.predict(test_x_)
  print(test_Y)
  print(predict_y)
  print(model.get_params())

def load_model():
  model_path = os.path.join(code_path,'LinearRegress.sav')
  return pickle.load(open(model_path, 'rb'))

def csv_read():
  print("Reading directory:",input_dir)
  files = os.listdir(input_dir)
  files.sort(key=lambda x:x[:11])
  print('Training Sequence:',len(files))
  Eps = []
  TrainX = []
  for filename in files:
    csv_path = os.path.join(input_dir,filename)
    # Eps = []
    # TrainX = []
    with open(csv_path,newline='') as csv_file:
      rows = csv.reader(csv_file)
      first = True
      for row in rows:
        if first:
          first = False
          continue
        if str(row).find('nan') != -1 or str(row).find('inf') != -1:
          continue
        Eps.append(float(row[0]))
        input_row = []
        input_row.append(float(row[4])) # vel
        input_row.append(float(row[5])) # r
        input_row.append(float(row[6])) # dt
        input_row.append(float(row[7])) # scan num
        # input_row.append(float(row[8])) # vx
        # input_row.append(float(row[9])) # vy
        # input_row.append(float(row[10])) # rcs
        # input_row.append(float(row[11])) # angle
        # input_row.append(float(row[12])) # vel_dir
        TrainX.append(input_row)
  EpsMatrix = np.asarray(Eps)
  EpsMatrix.reshape(len(Eps),1)
  Eps_min = np.min(EpsMatrix)
  Eps_max = np.max(EpsMatrix)
  EpsMatrix = (EpsMatrix-Eps_min)/(Eps_max-Eps_min)
  TrainMatrix = np.asarray(TrainX)
  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(np.array(TrainMatrix), np.array(EpsMatrix), test_size=0.2)
  print('Training Element: vel, r, dt, scan num')
  train_model(Y_train,X_train,Y_test,X_test)
  return

if __name__ == '__main__':
  # csv_read()
  model = load_model()
  code = m2c.export_to_c(model)
  print(code)