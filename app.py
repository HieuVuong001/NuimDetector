from flask import Flask, request, send_file, render_template
from PIL import Image, ImageDraw
import io
from ultralytics import YOLO

# Initialize the Flask application
app = Flask(__name__)

# Load your YOLOv5 model
model = YOLO('/home/jv/models/train/weights/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    # Load the uploaded image
    img = Image.open(request.files['image']).convert('RGB')

    # Run the model inference
    results = model(img, save=True)

    # Draw bounding boxes on the original image
    # draw = ImageDraw.Draw(img)
    # for row in results.pandas().xyxy[0].itertuples():
    #     draw.rectangle([row.xmin, row.ymin, row.xmax, row.ymax], outline="red", width=3)

    img = Image.open('./runs/detect/predict/image0.jpg')
    # Save to an in-memory byte stream
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)

    # Return the processed image
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
