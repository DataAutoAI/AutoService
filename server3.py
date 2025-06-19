from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from autogluon.tabular import TabularPredictor
from autogluon.features import AutoMLPipelineFeatureGenerator
import pandas as pd
import json
import numpy as np
import os
import ray
import time
import uuid
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

                
        # Handle features
        feature_generator = AutoMLPipelineFeatureGenerator()
        feature_generator.fit(train_data.drop(columns=[label_name]))
       
       
        model_path_id = request.form.get('model_id')
        result_model_id = model_path_id
        if not model_path_id: 
            model_id = uuid.uuid4()
            result_model_id = model_id
            model_path = f'models/ag_model_{model_id}'
        else :
            model_path = f'models/ag_model_{model_path_id}'
        
        # Ensure model_path directory can be created
        os.makedirs(os.path.dirname(model_path) or 'models', exist_ok=True)

        if not model_path:
            predictor = TabularPredictor(
                label=label_name            )
        else :
             predictor = TabularPredictor(
                label=label_name,
                path=model_path
            )
        # Train model
    
        predictor.fit(
            train_data,
            presets='best',
            time_limit=15, 
            hyperparameter_tune_kwargs='auto',dynamic_stacking=False, num_stack_levels=1
        )
        print("flag check this")
        # Get model information

        os.makedirs('models', exist_ok=True)
        with open('models/model_info.json', 'w') as f:
            json.dump({
                'model_path': model_path,
                'label_name': label_name,
                'last_updated': time.strftime("%Y-%m-%d %H:%M:%S")
            }, f)


        model_info = {
            'best_model': predictor.model_best,
            'model_names': predictor.model_names(),
            'leaderboard': predictor.leaderboard().to_dict(orient='records'),
            'train_samples': len(train_data),
            'features': list(train_data.columns),
            'model_path': predictor.path,
            'model_id': result_model_id ,
            'total_time': predictor.fit_summary().get('total_time', 'Unknown'),
            'train_accuracy': predictor.evaluate(train_data).get('accuracy', 'Unknown')
        }
        
        # Shutdown Ray to avoid conflicts
        ray.shutdown()
        
        return jsonify({
            'message': 'Model trained successfully',
            'model_info': model_info
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-model', methods=['POST'])
def test_model():
    try:
        # Get model_id from query parameter
      
        model_id = request.form.get('model_id')
        if not model_id :
            return jsonify({
            'message': 'model_id is not empty',
        }), 400 

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
        
        return jsonify({
            'message': 'Prediction completed successfully',
        }), 200
        
    except Exception as e:
        if ray.is_initialized():
            ray.shutdown()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)