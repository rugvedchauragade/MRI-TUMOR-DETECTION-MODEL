import os 
import numpy as np 
from flask import Flask, render_template ,request, send_from_directory
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img , img_to_array

# initialize flask app 
app = Flask(__name__)

# load model from the same directory as main.py 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR , 'brain_tumor_vgg16_model.h5')

print(f"Loading VGG16 model from : {MODEL_PATH}")
model = load_model(MODEL_PATH)

CLASS_NAMES = ['glioma' , 'meningioma' ,'notumor' ,'pituitary']

# upload folder configuration 
UPLOAD_FOLDER = os.path.join(BASE_DIR , 'static' , 'uploads')
os.makedirs(UPLOAD_FOLDER , exist_ok = True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'] , filename)

def predict_tumor(image_path):
    # preprocessing image for vgg16 (128x128 , normalized)
    img = load_img(image_path , target_size = (128,128))
    img_array= img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array , axis =0)

    # model interference 
    predictions =model.predict(img_batch)[0]
    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx])*100

    # class probability dictionary 
    probabilities = {
        name.capitalize(): round(float(prob) * 100 , 2)
        for name , prob in zip(CLASS_NAMES , predictions)
    }

    if predicted_class == 'notumor' :
        result_title = "NO TUMOR DETECTION"
        badge_style = "bg-success"
    else :
        result_title = f"TUMOR DETECTION : {predicted_class.upper()}"
        badge_style = "bg-danger"

    return result_title , round(confidence , 2) , probabilities , badge_style

@app.route('/' , methods = ['GET' , 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html')

        if file :
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'] , filename)
            file.save(file_path)

            result_title , confidence , probabilities , badge_style = predict_tumor(file_path)
            image_url = f"/static/uploads/{filename}"

            return render_template('index.html',
                                    result = result_title , 
                                    confidence = confidence ,
                                    probabilities = probabilities ,
                                    badge_style = badge_style ,
                                    image_url = image_url )

    return render_template ('index.html')

if __name__ =='__main__':
    app.run(debug=True)
                                
