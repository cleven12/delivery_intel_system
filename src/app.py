import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from ml_models import DeliveryPredictor, AnomalyDetector

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "hackathon-delivery-intelligence-2024")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 250 * 1024 * 1024  # 250MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables to store processed data
processed_data = None
prediction_results = None
anomaly_results = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page with file upload"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload and processing"""
    global processed_data, prediction_results, anomaly_results
    
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename or 'uploaded_file.csv')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded file
            app.logger.info(f"Processing file: {filename}")
            processor = DataProcessor()
            processed_data = processor.load_and_clean_data(filepath)
            
            if processed_data is None or len(processed_data) == 0:
                flash('Error processing file: No valid data found', 'error')
                return redirect(url_for('index'))
            
            # Train prediction model
            app.logger.info("Training prediction model...")
            predictor = DeliveryPredictor()
            predictor.train(processed_data)
            prediction_results = predictor.predict(processed_data)
            
            # Detect anomalies
            app.logger.info("Detecting anomalies...")
            anomaly_detector = AnomalyDetector()
            anomaly_results = anomaly_detector.detect_anomalies(processed_data)
            
            flash(f'File uploaded and processed successfully! {len(processed_data)} records processed.', 'success')
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            app.logger.error(f"Error processing file: {str(e)}")
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload a CSV file.', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Display dashboard with comprehensive error handling"""
    global processed_data, prediction_results, anomaly_results
    
    try:
        if processed_data is None or len(processed_data) == 0:
            flash('Please upload a dataset first', 'warning')
            return redirect(url_for('index'))
        
        # Calculate summary statistics with error handling
        try:
            total_deliveries = len(processed_data)
            avg_delivery_time = float(np.mean(processed_data['actual_delivery_time'])) if 'actual_delivery_time' in processed_data.columns else 0.0
            total_anomalies = len(anomaly_results.get('anomalies', [])) if anomaly_results and isinstance(anomaly_results, dict) else 0
            accuracy_score = float(prediction_results.get('accuracy', 0)) if prediction_results and isinstance(prediction_results, dict) else 0.0
        except Exception as e:
            app.logger.error(f"Error calculating dashboard statistics: {str(e)}")
            total_deliveries = 0
            avg_delivery_time = 0.0
            total_anomalies = 0
            accuracy_score = 0.0
        
        # Prepare data for charts with error handling
        try:
            chart_data = prepare_chart_data()
        except Exception as e:
            app.logger.error(f"Error preparing chart data: {str(e)}")
            chart_data = {}
        
        return render_template('dashboard.html',
                             total_deliveries=total_deliveries,
                             avg_delivery_time=round(avg_delivery_time, 2),
                             total_anomalies=total_anomalies,
                             accuracy_score=round(accuracy_score * 100, 2),
                             chart_data=chart_data,
                             anomaly_data=anomaly_results or {})
        
    except Exception as e:
        app.logger.error(f"Critical error in dashboard route: {str(e)}")
        flash('An error occurred while loading the dashboard', 'error')
        return redirect(url_for('index'))

@app.route('/api/chart-data')
def get_chart_data():
    """API endpoint to get chart data for JavaScript with error handling"""
    try:
        chart_data = prepare_chart_data()
        return jsonify(chart_data)
    except Exception as e:
        app.logger.error(f"Error in chart data API: {str(e)}")
        return jsonify({'error': 'Failed to prepare chart data'}), 500

def prepare_chart_data():
    """Prepare data for Chart.js visualizations with comprehensive error handling"""
    global processed_data, prediction_results, anomaly_results
    
    try:
        # Default empty response structure
        empty_response = {
            'actual_vs_predicted': {},
            'city_analysis': {},
            'anomaly_distribution': {'normal': 0, 'anomalies': 0}
        }
        
        if processed_data is None or len(processed_data) == 0:
            app.logger.warning("No processed data available for chart preparation")
            return empty_response
        
        # Handle case where prediction_results might be None or empty
        if prediction_results is None or not isinstance(prediction_results, dict):
            app.logger.warning("Prediction results is None or invalid format")
            return {
                'actual_vs_predicted': {},
                'city_analysis': {},
                'anomaly_distribution': {'normal': len(processed_data), 'anomalies': 0}
            }
        
        predictions = prediction_results.get('predictions')
        if predictions is None:
            app.logger.warning("No predictions found in prediction results")
            return {
                'actual_vs_predicted': {},
                'city_analysis': {},
                'anomaly_distribution': {'normal': len(processed_data), 'anomalies': 0}
            }
        
        # Convert predictions to list if it's numpy array
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        
        if not isinstance(predictions, (list, np.ndarray)) or len(predictions) == 0:
            app.logger.warning("Invalid predictions format or empty predictions")
            return {
                'actual_vs_predicted': {},
                'city_analysis': {},
                'anomaly_distribution': {'normal': len(processed_data), 'anomalies': 0}
            }
        
    except Exception as e:
        app.logger.error(f"Error in chart data preparation initialization: {str(e)}")
        return {
            'actual_vs_predicted': {},
            'city_analysis': {},
            'anomaly_distribution': {'normal': 0, 'anomalies': 0}
        }
    
    try:
        # For large datasets, use smart sampling for chart performance
        data_size = len(processed_data)
        prediction_size = len(predictions)
        
        # Scale sample size based on dataset size for better representation
        if data_size <= 1000:
            sample_size = min(50, data_size, prediction_size)
        elif data_size <= 10000:
            sample_size = min(100, data_size, prediction_size)  
        elif data_size <= 50000:
            sample_size = min(200, data_size, prediction_size)
        else:
            # For very large datasets (200MB+), use larger sample but still manageable
            sample_size = min(500, data_size, prediction_size)
        
        app.logger.info(f"Using sample size {sample_size} from {data_size} total records for charts")
        
        if sample_size == 0:
            app.logger.warning("Sample size is 0, returning empty chart data")
            return {
                'actual_vs_predicted': {},
                'city_analysis': {},
                'anomaly_distribution': {'normal': len(processed_data), 'anomalies': 0}
            }
        
        # For large datasets, use stratified sampling for better representation
        if data_size > 1000:
            # Use random sampling spread across the dataset
            np.random.seed(42)  # For reproducible sampling
            sample_indices = sorted(np.random.choice(data_size, sample_size, replace=False))
        else:
            # Use sequential sampling for small datasets
            sample_indices = list(range(sample_size))
        
        # Safe data sampling with error handling
        try:
            sample_data = processed_data.iloc[sample_indices]
        except (IndexError, KeyError) as e:
            app.logger.error(f"Error sampling processed data: {str(e)}")
            return {
                'actual_vs_predicted': {},
                'city_analysis': {},
                'anomaly_distribution': {'normal': 0, 'anomalies': 0}
            }
        
        try:
            sample_predictions = [predictions[i] for i in sample_indices if i < len(predictions)]
            if len(sample_predictions) != sample_size:
                app.logger.warning(f"Prediction sampling mismatch: expected {sample_size}, got {len(sample_predictions)}")
                sample_size = len(sample_predictions)
        except (IndexError, TypeError) as e:
            app.logger.error(f"Error sampling predictions: {str(e)}")
            return {
                'actual_vs_predicted': {},
                'city_analysis': {},
                'anomaly_distribution': {'normal': 0, 'anomalies': 0}
            }
        
    except Exception as e:
        app.logger.error(f"Error in data sampling: {str(e)}")
        return {
            'actual_vs_predicted': {},
            'city_analysis': {},
            'anomaly_distribution': {'normal': 0, 'anomalies': 0}
        }
    
    # Prepare data for actual vs predicted chart
    actual_vs_predicted = {
        'labels': [f'Delivery {i+1}' for i in range(sample_size)],
        'actual': sample_data['actual_delivery_time'].tolist(),
        'predicted': sample_predictions
    }
    
    # Prepare data for city-wise analysis
    city_analysis = {}
    if 'city' in processed_data.columns:
        cities = processed_data['city'].unique()[:5]  # Top 5 cities
        city_data = []
        for city in cities:
            city_mask = processed_data['city'] == city
            city_deliveries = processed_data[city_mask]
            if len(city_deliveries) > 0:
                avg_time = np.mean(city_deliveries['actual_delivery_time'])
                city_data.append(avg_time)
            else:
                city_data.append(0)
        
        if len(cities) > 0 and len(city_data) > 0:
            city_analysis = {
                'labels': cities.tolist(),
                'data': city_data
            }
    
    # Prepare anomaly data
    anomaly_count = len(anomaly_results['anomalies']) if anomaly_results and 'anomalies' in anomaly_results else 0
    anomaly_data = {
        'normal': len(processed_data) - anomaly_count,
        'anomalies': anomaly_count
    }
    
    return {
        'actual_vs_predicted': actual_vs_predicted,
        'city_analysis': city_analysis,
        'anomaly_distribution': anomaly_data
    }

@app.route('/anomalies')
def view_anomalies():
    """View detailed anomaly information"""
    global anomaly_results
    
    if anomaly_results is None:
        flash('No anomaly data available', 'warning')
        return redirect(url_for('dashboard'))
    
    return render_template('dashboard.html', show_anomalies=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
