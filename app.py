from flask import Flask , render_template , url_for, request
import numpy as np
import pandas as pd
import joblib
import pickle

from tensorflow.python.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input 
import numpy as np
from torch import classes

model=load_model('model_vgg16.h5')
root='D:/STUDY FILES/Breast_Cancer_prediction-main/chest_xray/val/'

def lungPredict(fileName):
    
    
    img=image.load_img(root+fileName,target_size=(224,224))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    img_data=preprocess_input(x)
    classes=model.predict(img_data)
    if(classes[0][0]):
        return 1
    else:
        return 0




app = Flask(__name__)
model2 = joblib.load('model_save2')
@app.route("/index.html")
def hello_world():
    # return "<p>Hello, World!</p>"
    return render_template('index.html')
@app.route("/About.html")
def About():
    return render_template('About.html')
@app.route("/cancer1.html" , methods=['POST','GET'])
def Cancer1():
    

    return render_template('cancer1.html')
@app.route("/cancer2.html", methods=['POST','GET'])
def Cancer2():
    return render_template('cancer2.html')
@app.route("/predict" , methods=['POST','GET'])
def predict():
    if(request.method=='POST'):
        
        radius_mean=float(request.form['radius_mean'])
        texture_mean=float(request.form['texture_mean']	)
        perimeter_mean=float(request.form['perimeter_mean']	)
        area_mean=float(request.form['area_mean'])
        smoothness_mean=float(request.form['smoothness_mean'])
        compactness_mean=float(request.form['compactness_mean']	)
        concavity_mean=float(request.form['concavity_mean'])
        concave_points_mean=float(request.form['concave_points_mean'])
        symmetry_mean=float(request.form['symmetry_mean'])
        fractal_dimension_mean=float(request.form['fractal_dimension_mean']	)
        radius_se=float(request.form['radius_se'])
        texture_se=float(request.form['texture_se'])
        perimeter_se=float(request.form['perimeter_se']	)
        area_se	=float(request.form['area_se'])
        smoothness_se=float(request.form['smoothness_se'])
        compactness_se=float(request.form['compactness_se'])
        concavity_se=float(request.form['concavity_se']	)
        concave_points_se=float(request.form['concave_points_se'])
        symmetry_se=float(request.form['symmetry_se'])
        fractal_dimension_se=float(request.form['fractal_dimension_se'])
        radius_worst=float(request.form['radius_worst']	)
        texture_worst=float(request.form['texture_worst'])
        perimeter_worst=float(request.form['perimeter_worst'])
        area_worst=float(request.form['area_worst'])
        smoothness_worst=float(request.form['smoothness_worst']	)
        compactness_worst=float(request.form['compactness_worst'])	
        concavity_worst=float(request.form['concavity_worst'])
        concave_points_worst=float(request.form['concave_points_worst']	)
        symmetry_worst=float(request.form['symmetry_worst']	)
        fractal_dimension_worst=float(request.form['fractal_dimension_worst'])

        patient=[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]
       # 

        
        patient=np.array([patient])
        predict=model2.predict(patient)[0]

        print("helo")
        if(int(predict)):
            return render_template('Benign.html')
        else:
            return  render_template('Malignant.html')
 
@app.route('/predict2',methods=['GET', 'POST'])
def predict2():

    if request.method == 'POST':
        image_file=request.files['image']
        location=image_file.filename
        if(lungPredict(image_file.filename)):
            return render_template('normalLung.html',image_loc=location)
        else:
            return render_template('infectedLung.html',image_loc=location)
    
    
    
if __name__ == "__main__":
    app.run(debug=True)