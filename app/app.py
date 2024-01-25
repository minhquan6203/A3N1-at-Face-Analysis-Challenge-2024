import os
from flask import Flask, render_template, request
from deep_learning_model import Face_Analysis
import json
import cv2
from PIL import Image
import time
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
            total_start_time = time.time()
            yolo_start_time = time.time()
            results = model.perform_yolo_detection(image_path=save_path)
            yolo_end_time = time.time()
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
            face_start_time = time.time()
            extracted_text = model.perform_face_classify([crop_img_path])
            face_end_time = time.time()
            
            total_end_time = time.time()
            total_execution_time = total_end_time - total_start_time
            yolo_execution_time = yolo_end_time - yolo_start_time
            face_execution_time = face_end_time - face_start_time

            print(f"Thời gian yolov8 thực hiện: {yolo_execution_time} s")
            print(f"Thời gian face analysis model thực hiện: {face_execution_time} s")
            print(f"Tổng thời gian thực hiện: {total_execution_time} s")
        except Exception as e:
            return render_template('error.html', error_message=str(e))
        return render_template('index.html', upload=True, uploaded_image=filename, text=extracted_text, croped_image=f"{filename.split('.')[0]}_{0}.jpg")

    return render_template('index.html', upload=False)

if __name__ == '__main__':
    app.run(debug=True)