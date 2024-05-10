from flask import Flask, request, send_file, render_template
from PIL import Image, ImageDraw
import torch
import io

# Initialize the Flask application
app = Flask(__name__)

# Load your YOLOv5 model
model_path = 'C:/Users/jeffr/Desktop/SJSU Course/24 Spring/CMPE-258/Project/finalModel.pt'
# model = ''
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, trust_repo=True)

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
    results = model(img)

    # Draw bounding boxes on the original image
    draw = ImageDraw.Draw(img)
    for row in results.pandas().xyxy[0].itertuples():
        draw.rectangle([row.xmin, row.ymin, row.xmax, row.ymax], outline="red", width=3)

    # Save to an in-memory byte stream
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)

    # Return the processed image
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
