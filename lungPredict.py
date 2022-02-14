



from tensorflow.python.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input 
import numpy as np
from torch import classes

model=load_model('model_vgg16.h5')
img=image.load_img('D:/STUDY FILES/Breast_Cancer_prediction-main/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg',target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
classes=model.predict(img_data)
if(classes[0][0]):
    print("normal person")
else:
    print("infected person")
#print(classes[0][1])