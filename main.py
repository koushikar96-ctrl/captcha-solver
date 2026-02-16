from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

app = Flask(__name__)

# ---------------- FOLDERS ----------------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- PARAMETERS ----------------
img_width, img_height = 128, 64
captcha_length = 5
characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Initialize LabelBinarizer exactly like during training
lb = LabelBinarizer()
lb.fit(list(characters))
classes = lb.classes_

# ---------------- LOAD CNN MODEL ----------------
cnn_model_path = r"models/captcha_cnn_model_v2.h5"
cnn_model = load_model(cnn_model_path)
print("CNN model loaded successfully.")

# ---------------- HELPER FUNCTIONS ----------------
def preprocess_image(img_path):
    """Read image, grayscale, resize, normalize, add batch & channel dims."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # (H,W,1)
    img = np.expand_dims(img, axis=0)   # (1,H,W,1)
    return img

def decode_prediction(pred):
    """Decode CNN prediction to text."""
    text = ""
    for i in range(pred.shape[0]):
        char_index = np.argmax(pred[i])
        text += classes[char_index]
    return text

def predict_captcha(image_path):
    """Preprocess uploaded image and predict text using CNN model."""
    img = preprocess_image(image_path)
    pred = cnn_model.predict(img, verbose=0)
    decoded_text = decode_prediction(pred[0])
    return decoded_text

# ---------------- FLASK ROUTE ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    prediction = ""
    if request.method == 'POST':
        if 'captcha_image' in request.files:
            file = request.files['captcha_image']
            if file.filename != '':
                # Secure the filename and save
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Predict using the CNN model
                prediction = predict_captcha(filepath)

    return render_template('index.html', filename=filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)