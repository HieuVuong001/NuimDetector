from flask import Flask, flash, Response,  request, send_file, render_template, redirect, url_for
from werkzeug.utils import secure_filename

from PIL import Image
import io
import os
try:
    __import__('YOLO')
except ImportError:
    os.system('pip install ultralytics')

from ultralytics import YOLO
import cv2

UPLOAD_FOLDER = './storage'

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your YOLOv5 model
model = YOLO('./models/yolov8_pretrained/weights/best.pt', verbose=False)

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

    print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    return redirect('/stream')


@app.route('/video')
def video_page():
    return render_template('video.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    # Load the uploaded image
    img = Image.open(request.files['image']).convert('RGB')

    # Run the model inference
    results = model(img, save=True, project='processed')

    # Draw bounding boxes on the original image
    # draw = ImageDraw.Draw(img)
    # for row in results.pandas().xyxy[0].itertuples():
    #     draw.rectangle([row.xmin, row.ymin, row.xmax, row.ymax], outline="red", width=3)

    img = Image.open(f'{results[0].save_dir}/image0.jpg')
    # Save to an in-memory byte stream
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)

    # Return the processed image and set flag to true
    ready = True
    print(ready)
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)