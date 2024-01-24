import os
from flask import Flask, render_template, request
from deep_learning_model import Face_Analysis
import json
import cv2
from PIL import Image
app = Flask(__name__)


BASE_PATH = os.getcwd()
UPLOAD_PATH =  'app/static/upload/'
CROPPED_PATH = 'app/static/predict/crop_face/'

model = Face_Analysis()

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        save_path = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(save_path)

        try:
            results = model.perform_yolo_detection(image_path=save_path)
            predictions=json.loads(results[0].tojson())
            img = Image.open(save_path).convert("RGB")
            for idx, prediction in enumerate(predictions):
                conf_score = prediction['confidence']
                if conf_score < 0.6:
                    continue
                bbox = prediction['box']
                xmin = int(bbox['x1'])
                ymin = int(bbox['y1'])
                xmax = int(bbox['x2'])
                ymax = int(bbox['y2'])
                cropped_img = img.crop([xmin, ymin, xmax, ymax])
                new_name=f"{filename.split('.')[0]}_{idx}.jpg"
                cropped_img.save(os.path.join(CROPPED_PATH,new_name))
            crop_img_path=os.path.join(CROPPED_PATH,f"{filename.split('.')[0]}_{0}.jpg")
            extracted_text = model.perform_face_classify([crop_img_path])
        except Exception as e:
            return render_template('error.html', error_message=str(e))
        return render_template('index.html', upload=True, uploaded_image=filename, text=extracted_text, croped_image=f"{filename.split('.')[0]}_{0}.jpg")

    return render_template('index.html', upload=False)

if __name__ == '__main__':
    app.run(debug=True)