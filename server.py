from flask import Flask
from flask import request
from flask import  jsonify
from werkzeug.utils import secure_filename
import os
from autogluon.tabular import TabularPredictor
import pandas as pd
import json
import numpy as np
from autogluon.features import AutoMLPipelineFeatureGenerator
from test2 import flatten_json

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        
    else:            
        # return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
        return jsonify({'error': 'File type not allowed'}), 400
    with open('/uploads/data.json', 'r') as file:
            train_json = json.load(file)

        # handle df data train
    train_data, label_name = flatten_json(train_json)
    print("Label name:", label_name)
    print("Train DataFrame:\n", train_data)


    # load data test file
    with open('test.json', 'r') as file:
        test_json = json.load(file)


        # handle df data test
    test_data, _ = flatten_json(test_json)
    print("Test DataFrame:\n", test_data)

    test_data = test_data.drop(columns=[label_name], errors='ignore')
    print("Test DataFrame (sau khi bỏ nhãn):\n", test_data)

    # hanlde feature 
    feature_generator = AutoMLPipelineFeatureGenerator()
    feature_generator.fit(train_data.drop(columns=[label_name]))
    feature_metadata = feature_generator.feature_metadata

    # predict
    predictor = TabularPredictor(
        label=label_name
    )
    predictor.fit(
        train_data,
        time_limit=600
    )


    # test
    predictions = predictor.predict(test_data)
    return jsonify({'data': predictions}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)