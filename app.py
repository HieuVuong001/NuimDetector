from flask import Flask, Response, request, send_file, render_template, redirect
from PIL import Image, ImageDraw
import io
from ultralytics import YOLO
import cv2
# Initialize the Flask application
app = Flask(__name__)

# Load your YOLOv5 model
model = YOLO('/home/jv/models/train/weights/best.pt', verbose=False)

global video

# flag for processed image
ready = False


video_path = '/home/jv/NuimDetector/14537338-hd_1920_1080_30fps.mp4'

global cap

@app.route('/')
def index():
    return render_template('index.html', ready=ready)


def generate_video():
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # cv2.imshow("Yolo8 Inference", annotated_frame)
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

    cap = cv2.VideoCapture(file.filename)

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
    app.run(debug=True, host='0.0.0.0', port=5000)