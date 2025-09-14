import numpy as np
import pandas as pd
import logging
import gc
from typing import Optional, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class DeliveryPredictor:
    """Machine learning model for predicting delivery completion time using NumPy"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.weights = None
        self.intercept = None
        self.feature_means = None
        self.feature_stds = None
        self.is_trained = False
    
    def train(self, data, batch_size: int = 10000):
        """Train linear regression model using NumPy with memory optimization for large datasets"""
        try:
            self.logger.info(f"Training delivery prediction model on {len(data)} records...")
            
            # For very large datasets, use sampling for training to manage memory
            if len(data) > 50000:
                self.logger.info(f"Large dataset detected ({len(data)} rows), using stratified sampling")
                # Sample 50k representative records for training
                sample_size = min(50000, len(data))
                data_sample = data.sample(n=sample_size, random_state=42)
                self.logger.info(f"Training on {len(data_sample)} sampled records")
            else:
                data_sample = data
            
            # Prepare features and target
            X, y = self._prepare_features(data_sample)
            
            if X is None or y is None:
                self.logger.error("Failed to prepare features for training")
                return False
            
            # Normalize features
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            X_normalized = (X - self.feature_means) / (self.feature_stds + 1e-8)
            
            # Add bias term
            X_with_bias = np.column_stack([np.ones(X_normalized.shape[0]), X_normalized])
            
            # Add debug information before training
            self.logger.info(f"Training data shapes - X: {X_normalized.shape}, y: {len(y)}")
            self.logger.info(f"Feature means: {self.feature_means}")
            self.logger.info(f"Feature stds: {self.feature_stds}")
            self.logger.info(f"X_normalized stats - mean: {np.mean(X_normalized, axis=0)}, std: {np.std(X_normalized, axis=0)}")
            
            # Train using normal equation: θ = (X^T X)^(-1) X^T y
            try:
                XtX = X_with_bias.T @ X_with_bias
                Xty = X_with_bias.T @ y
                
                # Check if matrix is well-conditioned
                condition_number = np.linalg.cond(XtX)
                self.logger.info(f"Design matrix condition number: {condition_number:.4f}")
                
                theta = np.linalg.solve(XtX, Xty)
                self.intercept = theta[0]
                self.weights = theta[1:]
                self.is_trained = True
                
                self.logger.info(f"Model weights: intercept={self.intercept:.4f}, weights={self.weights}")
                self.logger.info("Model training completed successfully")
                return True
                
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if matrix is singular
                self.logger.info("Matrix is singular, using pseudo-inverse")
                theta = np.linalg.pinv(X_with_bias) @ y
                self.intercept = theta[0]
                self.weights = theta[1:]
                self.is_trained = True
                
                self.logger.info(f"Model weights: intercept={self.intercept:.4f}, weights={self.weights}")
                self.logger.info("Model training completed using pseudo-inverse")
                return True
                
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            return False
    
    def predict(self, data):
        """Make predictions on new data"""
        if not self.is_trained:
            self.logger.error("Model is not trained yet")
            return None
        
        try:
            X, y_true = self._prepare_features(data)
            
            if X is None:
                self.logger.error("Failed to prepare features for prediction")
                return None
            
            # Normalize features
            if self.feature_means is not None and self.feature_stds is not None:
                X_normalized = (X - self.feature_means) / (self.feature_stds + 1e-8)
            else:
                self.logger.error("Feature means or stds are None, cannot normalize")
                return None
            
            # Make predictions
            y_pred = self.intercept + X_normalized @ self.weights
            
            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy(y_true, y_pred)
            
            results = {
                'predictions': y_pred,
                'actual': y_true,
                'accuracy': accuracy_metrics['r2'],
                'mae': accuracy_metrics['mae'],
                'rmse': accuracy_metrics['rmse']
            }
            
            self.logger.info(f"Predictions completed. R² Score: {accuracy_metrics['r2']:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            return None
    
    def _prepare_features(self, data):
        """Prepare feature matrix and target vector"""
        try:
            # Select features for prediction - prioritize basic numeric features
            feature_columns = []
            
            # Start with reliable basic numeric features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            excluded_cols = ['actual_delivery_time']
            basic_features = [col for col in numeric_cols if col not in excluded_cols]
            
            self.logger.info(f"Available numeric columns: {list(numeric_cols)}")
            
            # Prioritize specific features that should have good variance
            preferred_features = ['package_weight', 'pickup_latitude', 'pickup_longitude', 
                                'delivery_latitude', 'delivery_longitude']
            
            for feature in preferred_features:
                if feature in basic_features:
                    feature_columns.append(feature)
            
            # Add any remaining basic features up to 5 total
            for feature in basic_features:
                if feature not in feature_columns and len(feature_columns) < 5:
                    feature_columns.append(feature)
            
            # Only use engineered features as fallback if no basic features
            if not feature_columns:
                self.logger.info("No basic numeric features found, trying engineered features")
                if 'distance' in data.columns:
                    feature_columns.append('distance')
                if 'hour_of_day' in data.columns:
                    feature_columns.append('hour_of_day')
                if 'package_weight' in data.columns:
                    feature_columns.append('package_weight')
                
            self.logger.info(f"Selected features for training: {feature_columns}")
            
            if not feature_columns:
                self.logger.warning("No valid features available for training")
            
            if not feature_columns:
                # Fallback: create basic features from available data
                feature_columns = []
                
                # Try to use package weight if available
                if 'package_weight' in data.columns:
                    feature_columns.append('package_weight')
                
                # Use coordinates if available
                coordinate_cols = [col for col in data.columns if 'latitude' in col.lower() or 'longitude' in col.lower()]
                for col in coordinate_cols[:4]:  # Use up to 4 coordinate features
                    feature_columns.append(col)
                
                # If still no features, create synthetic ones
                if not feature_columns:
                    feature_columns = ['synthetic_feature_1', 'synthetic_feature_2']
                    np.random.seed(42)
                    data = data.copy()
                    data['synthetic_feature_1'] = np.random.normal(0, 1, len(data))
                    data['synthetic_feature_2'] = np.random.normal(0, 1, len(data))
                    self.logger.warning("Using synthetic features as fallback")
            
            # Prepare target vector first to ensure it exists
            if 'actual_delivery_time' not in data.columns:
                self.logger.error("Target variable 'actual_delivery_time' not found")
                return None, None
            
            y = data['actual_delivery_time'].values.astype(float)
            
            # Prepare feature matrix
            try:
                X_df = data[feature_columns].copy()
                
                # Fill NaN values with median for numeric columns
                for col in feature_columns:
                    if col in X_df.columns:
                        median_val = X_df[col].median()
                        if pd.isna(median_val):
                            # If all values are NaN, use a default
                            X_df[col].fillna(1.0, inplace=True)
                        else:
                            X_df[col].fillna(median_val, inplace=True)
                
                X = X_df.values.astype(float)
                
            except (KeyError, ValueError) as e:
                self.logger.error(f"Error creating feature matrix: {str(e)}")
                # Try to handle missing columns
                valid_features = [col for col in feature_columns if col in data.columns]
                if not valid_features:
                    self.logger.error("No valid feature columns found")
                    return None, None
                
                X_df = data[valid_features].copy()
                
                # Fill NaN values for valid features
                for col in valid_features:
                    median_val = X_df[col].median()
                    if pd.isna(median_val):
                        X_df[col].fillna(1.0, inplace=True)
                    else:
                        X_df[col].fillna(median_val, inplace=True)
                
                X = X_df.values.astype(float)
                feature_columns = valid_features
            
            # Only remove rows if y has NaN values
            valid_indices = ~np.isnan(y)
            X = X[valid_indices]
            y = y[valid_indices]
            
            self.logger.info(f"Prepared features: {feature_columns}")
            self.logger.info(f"Feature matrix shape: {X.shape}")
            self.logger.info(f"Valid samples: {len(X)} out of {len(data)}")
            
            if len(X) == 0:
                self.logger.error("No valid samples after removing NaN values")
                return None, None
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None
    
    def _calculate_accuracy(self, y_true, y_pred):
        """Calculate accuracy metrics"""
        try:
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            
            # Calculate R² score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Add debug logging
            self.logger.info(f"Accuracy calculation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            self.logger.info(f"Target stats - mean: {np.mean(y_true):.4f}, std: {np.std(y_true):.4f}, range: {np.min(y_true):.4f}-{np.max(y_true):.4f}")
            self.logger.info(f"Prediction stats - mean: {np.mean(y_pred):.4f}, std: {np.std(y_pred):.4f}, range: {np.min(y_pred):.4f}-{np.max(y_pred):.4f}")
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
        except Exception as e:
            self.logger.error(f"Error calculating accuracy: {str(e)}")
            return {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0}


class AnomalyDetector:
    """Anomaly detection using statistical methods"""
    
    def __init__(self, threshold_std=2.5):
        self.logger = logging.getLogger(__name__)
        self.threshold_std = threshold_std
    
    def detect_anomalies(self, data):
        """Detect anomalies in delivery times using statistical methods"""
        try:
            self.logger.info("Detecting anomalies in delivery data...")
            
            if 'actual_delivery_time' not in data.columns:
                self.logger.error("Target variable 'actual_delivery_time' not found")
                return {'anomalies': [], 'statistics': {}}
            
            delivery_times = data['actual_delivery_time'].values
            
            # Calculate statistics
            mean_time = np.mean(delivery_times)
            std_time = np.std(delivery_times)
            median_time = np.median(delivery_times)
            
            # Z-score method for anomaly detection
            z_scores = np.abs((delivery_times - mean_time) / std_time)
            anomaly_mask = z_scores > self.threshold_std
            
            # Modified Z-score using median (more robust to outliers)
            mad = np.median(np.abs(delivery_times - median_time))
            modified_z_scores = 0.6745 * (delivery_times - median_time) / mad if mad != 0 else np.zeros_like(delivery_times)
            modified_anomaly_mask = np.abs(modified_z_scores) > self.threshold_std
            
            # Combine both methods
            combined_anomaly_mask = anomaly_mask | modified_anomaly_mask
            
            # Extract anomalies
            anomaly_indices = np.where(combined_anomaly_mask)[0]
            anomalies = []
            
            for idx in anomaly_indices:
                anomaly_record = {
                    'index': int(idx),
                    'actual_time': float(delivery_times[idx]),
                    'z_score': float(z_scores[idx]),
                    'deviation_from_mean': float(delivery_times[idx] - mean_time),
                    'anomaly_type': self._classify_anomaly(delivery_times[idx], mean_time, std_time)
                }
                
                # Add additional context if available
                if 'city' in data.columns:
                    anomaly_record['city'] = data.iloc[idx]['city']
                
                anomalies.append(anomaly_record)
            
            # Calculate statistics
            statistics = {
                'total_records': len(data),
                'total_anomalies': len(anomalies),
                'anomaly_percentage': (len(anomalies) / len(data)) * 100,
                'mean_delivery_time': float(mean_time),
                'std_delivery_time': float(std_time),
                'median_delivery_time': float(median_time),
                'threshold_used': self.threshold_std
            }
            
            self.logger.info(f"Anomaly detection completed. Found {len(anomalies)} anomalies ({statistics['anomaly_percentage']:.2f}%)")
            
            return {
                'anomalies': anomalies,
                'statistics': statistics
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            return {'anomalies': [], 'statistics': {}}
    
    def _classify_anomaly(self, value, mean, std):
        """Classify the type of anomaly"""
        if value > mean + 2 * std:
            return 'Unusually Long Delivery'
        elif value < mean - 2 * std:
            return 'Unusually Short Delivery'
        else:
            return 'Moderate Anomaly'
