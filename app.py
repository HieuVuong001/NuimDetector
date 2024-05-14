from flask import Flask, flash, Response,  request, send_file, render_template, redirect, url_for
from werkzeug.utils import secure_filename

from PIL import Image
import os
import requests
from io import BytesIO

try:
    __import__('ultralytics')
except ImportError:
    os.system('pip install ultralytics')

from ultralytics import YOLO
import cv2

UPLOAD_FOLDER = './storage'

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your YOLOv5 model
model = YOLO('./trained_models/yolov8_pretrained/weights/best.pt', verbose=False)

global cap

@app.route('/')
def index():
    return render_template('index.html')

def generate_video():
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Break the loop if the end of the video is reached
            break

@app.route('/stream')
def stream():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict_video', methods=['POST'])
def predict_video():
    global cap
    # Get video path in request
    file = request.files['video']

    filename = secure_filename(file.filename)

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    cap =  cv2.VideoCapture(f'{UPLOAD_FOLDER}/{filename}')

    return redirect('/stream')


@app.route('/video')
def video_page():
    return render_template('video.html')



@app.route('/predict', methods=['POST'])
def predict():
    # if 'image' not in request.files:
    #     return "No image uploaded", 400
    if 'img_src' in request.form:
        # Get the image from within the site
        response = requests.get(request.form['img_src'])
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        # Load the uploaded image
        img = Image.open(request.files['image']).convert('RGB')

    # Run the model inference
    results = model(img, save=True, project='processed', verbose=False)

    
    img = Image.open(f'{results[0].save_dir}/image0.jpg')
    
    # Save to an in-memory byte stream
    img_io = BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)

    # Return the processed image and set flag to true
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)