from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from model import Predict

import numpy as np
import matplotlib.pyplot as plt
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'jfif'}

# Создаем папку для загрузок
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
predictor = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    global predictor

    if request.method == 'POST':
        # Обработка загрузки файла
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Создаем экземпляр Predict и обрабатываем изображение
            threshold = float(request.form.get('threshold', 0.6))
            predictor = Predict(filepath, score_threshold=threshold)
            predictor.find()
            num_objects = predictor.valide()
            
            return render_template('index.html', 
                                 image_url=url_for('static', filename=f'uploads/{filename}'),
                                 threshold=threshold,
                                 num_objects=num_objects)
    
    return render_template('index.html')

@app.route('/process_click', methods=['POST'])
def process_click():
    global predictor

    data = request.json
    # image_path = os.path.join('static', data['image_url'].lstrip('/'))
    image_path = data['image_url'].lstrip('/')
    x, y = data['x'], data['y']
    threshold = float(data['threshold'])

    print(image_path)
    # predictor = Predict(image_path, score_threshold=threshold)
    # predictor.find()
    # predictor.valide()

    result_image = predictor.point(y, x)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_result.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
    return jsonify({
        'result_url': url_for('static', filename='uploads/temp_result.jpg')
    })

if __name__ == '__main__':
    app.run(debug=True)