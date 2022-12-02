import numpy as np                                                      # import numpy
import tensorflow as tf                                                 # import package for ML algorithms

image_size = 224                                                        # varaible of size of the image
classes = ["LUNG_CANCER","NOT_LUNG_CANCER"]

def ensemble(x, weights, models):                                       # function to do ensemble learning of models
  '''
  returns a weighted average of predictions made by the models\n
  x -> input image \n
  weights -> a list of weights \n
  models -> a list of models\n    
  '''      
  outputs = []    
  for model in models:
      outputs.append(list(model.predict(x)[0]))                

  outputs = np.array(outputs)
  avg = np.average(a=outputs,axis=0,weights=weights)                  # average of each models
  return avg


def equal(pred, label):                                                 # check if prediction is equal
  pred_id = np.argmax(pred)
  if (pred_id == label):
    return True
  else:
    return False


def accuracy(predicted_values, y_truths):                               # function to get accurcy of each model
  '''
  returns accuracy\n
  predicted_values = a numpy array containing the predictions\n
  y_truths = a numpy array containing the truth values\n  
  '''
  total = len(y_truths)
  correct = 0
  for i in range(len(y_truths)):
    if equal(predicted_values[i],y_truths[i]):
      correct += 1
  acc = correct / total
  return acc


def generate_weights(x_val, y_val, models):                             # function to generate weights for models
  '''
  returns a list of weights
  '''
  accuracy = []
  weights = np.full((1,len(models)), 100.0)
  for model in models:
    acc = model.evaluate(x_val, y_val)[1]
    accuracy.append(100*acc)
  weights = weights - accuracy
  weights = weights**2
  sum = np.sum(weights)
  weights = weights/sum
  weights = 1/weights
  weights = weights**2
  sum = np.sum(weights)
  weights = weights/sum
  return weights

#%% [code]
import tensorflow as tf 
import numpy as np
# from Ensembling import *
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm_notebook

# paths of training and test npy files and trained model files
x_test_path = '/content/drive/MyDrive/MLFPGA/MLFPGA_Proj/DATA/OUTPUT_DIR/New_test_images.npy'
y_test_path = '/content/drive/MyDrive/MLFPGA/MLFPGA_Proj/DATA/OUTPUT_DIR/New_test_labels.npy'
inception_path = '/content/drive/MyDrive/MLFPGA/MLFPGA_Proj/Model_Data_all_models/inception_v3.h5'
resnet_path = '/content/drive/MyDrive/MLFPGA/MLFPGA_Proj/Model_Data_all_models/resnet50_v2.h5'
densenet_path = '/content/drive/MyDrive/MLFPGA/MLFPGA_Proj/Model_Data_all_models/densenet201.h5'
# vgg19_path = '/content/drive/MyDrive/MLFPGA/MLFPGA_Proj/Model_Data_all_models/vgg19.h5'
mobilenet_path = '/content/drive/MyDrive/MLFPGA/MLFPGA_Proj/Model_Data_all_models/mobilenet_v2.h5'
xception_path = '/content/drive/MyDrive/MLFPGA/MLFPGA_Proj/Model_Data_all_models/xception.h5'

# load test data
x_test = np.load(x_test_path)
y_test = np.load(y_test_path)

image_size = 224

# load each of the models
inception_model = tf.keras.models.load_model(inception_path)
resnet_model = tf.keras.models.load_model(resnet_path)
densenet_model = tf.keras.models.load_model(densenet_path)
# vgg19_model = tf.keras.models.load_model(vgg19_path)
mobilenet_model = tf.keras.models.load_model(mobilenet_path)
xception_model = tf.keras.models.load_model(xception_path)

models = [densenet_model,resnet_model,mobilenet_model,xception_model, inception_model]
# generate weight for each model
print('Generating weights...')
w = generate_weights(x_test,y_test,models)[0] #generating weights
print("Weights: ", w)

predictions = []
print("Performing Ensemble of the 3 models...")
for i in tqdm_notebook(range(len(x_test))):
  pred = ensemble(x_test[i].reshape(-1,image_size,image_size,3),w,models) # do ensemble learning
  predictions.append(pred)

print("Accuracy: ",round(accuracy(predictions,y_test),2))

y_pred = np.argmax(np.array(predictions), axis=1)
# y_pred = [np.argmax(prediction[0]) for prediction in predictions]

print("The classification report: ")
print(classification_report(y_pred=y_pred, y_true=y_test))              # print classification report
print()
print("Confusion Matrix: ")
print(confusion_matrix(y_pred=y_pred, y_true=y_test))                   # print confusion matrix