from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from autogluon.tabular import TabularPredictor
from autogluon.features import AutoMLPipelineFeatureGenerator
import pandas as pd
import json
import numpy as np
import os
import ray

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def flatten_json(json_data, id_field=None):
    """
    Làm phẳng JSON động thành DataFrame.
    Args:
        json_data: Dict từ JSON.
        id_field: Tên cột ID (None để lấy từ key của "data").
    Returns:
        Tuple (DataFrame, label_name).
    """
    records = []
    data = json_data.get('data', {})
    
    # Lấy label_name từ "lable"
    label_name = json_data.get('lable', None)
    
    # Nếu label_name rỗng, suy ra từ "label" trong "data"
    if not label_name or not isinstance(label_name, str) or label_name.strip() == '':
        if isinstance(data, dict) and data:
            first_item = next(iter(data.values()))
            label_dict = first_item.get('label', {})
            if label_dict and isinstance(label_dict, dict):
                label_name = next(iter(label_dict), 'label')
        elif isinstance(data, list) and data:
            label_dict = data[0].get('label', {})
            if label_dict and isinstance(label_dict, dict):
                label_name = next(iter(label_dict), 'label')
        else:
            label_name = 'label'
    
    # Case 1: "data" là object {}
    if isinstance(data, dict):
        for key, value in data.items():
            record = {id_field or 'id': key} if id_field or 'id' else {}
            feature = value.get('feature', {})
            record.update(feature)
            label = value.get('label', {})
            record[label_name] = label.get(label_name, np.nan)
            records.append(record)
    
    # Case 2: "data" là mảng []
    elif isinstance(data, list):
        for i, value in enumerate(data, 1):
            record = {id_field or 'id': i} if id_field or 'id' else {}
            feature = value.get('feature', {})
            record.update(feature)
            label = value.get('label', {})
            record[label_name] = label.get(label_name, np.nan)
            records.append(record)
    
    df = pd.DataFrame(records)
    # Chuyển đổi kiểu dữ liệu
    for col in df.columns:
        if col != (id_field or 'id') and col != label_name:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col])
    
    return df, label_name

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed, only JSON allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read uploaded train JSON
        with open(file_path, 'r') as file:
            train_json = json.load(file)
        
        # Handle train data
        train_data, label_name = flatten_json(train_json)
        print("Label name:", label_name)
        print("Train DataFrame:\n", train_data)

        
        # Read test JSON (assume test.json exists)
        # with open('test.json', 'r') as file:
        #     test_json = json.load(file)
        
        # # Handle test data
        # test_data, _ = flatten_json(test_json)
        # print("Test DataFrame:\n", test_data)
        
        # test_data = test_data.drop(columns=[label_name], errors='ignore')
        # print("Test DataFrame (sau khi bỏ nhãn):\n", test_data)
        
        # Handle features
        feature_generator = AutoMLPipelineFeatureGenerator()
        feature_generator.fit(train_data.drop(columns=[label_name]))
        
        # Train model
        predictor = TabularPredictor(
            label=label_name,
        )
        predictor.fit(
            train_data,
            presets='best',
            time_limit=3600,  # 1 hour for max accuracy
            hyperparameter_tune_kwargs='auto',
        )
        print("flag check this")
        # Get model information
        model_info = {
            'best_model': predictor.model_best,
            'model_names': predictor.model_names,
            'leaderboard': predictor.leaderboard().to_dict(orient='records'),
            'train_samples': len(train_data),
            'features': list(train_data.columns),
            'model_path': predictor.path,
            'total_time': predictor.fit_summary().get('total_time', 'Unknown'),
            'train_accuracy': predictor.evaluate(train_data)['accuracy'],
            'best_model_params': predictor.info().get('model_info', {}).get(predictor.get_model_best(), {}).get('hyperparameters', {}),
            'data_summary': {
                'num_samples': len(train_json['data']),
                'label_name': label_name,
                'feature_columns': [col for col in train_data.columns if col != label_name and col != 'id'],
                'sample_data': list(train_json['data'].values())[:3]  # Return first 3 samples
            }
        }
        
        # Shutdown Ray to avoid conflicts
        ray.shutdown()
        
        return jsonify({
            'message': 'Model trained successfully',
            'model_info': model_info
        }), 200
    #     predictor.fit(
    #         train_data,
    #         presets='best',
    #         time_limit=600,  # 10 minutes
    #     )
        
    #     # Predict
    #     # predictions = predictor.predict(test_data)
        
    #     # Prepare response
    #     # output = pd.DataFrame({'id': test_data['id'], label_name: predictions})
    # # Get model information
    #     model_info = {
    #         'best_model': predictor.model_best,
    #         'model_names': predictor.model_names,
    #         'leaderboard': predictor.leaderboard().to_dict(orient='records'),
    #         'train_samples': len(train_data),
    #         'features': list(train_data.columns),
    #         'model_path': predictor.path,
    #         'fit_summary': predictor.fit_summary().get('total_time', 'Unknown'),
    #         'data':train_json
    #     }
    #     return jsonify({
    #         'message': 'File processed successfully',
    #         'model_info': model_info
    #         # 'predictions': output.to_dict(orient='records')
    #     }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)