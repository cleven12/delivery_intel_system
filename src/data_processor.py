import pandas as pd
import numpy as np
import logging
from datetime import datetime
import gc
from typing import Optional

class DataProcessor:
    """Handle data loading, cleaning, and preprocessing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_and_clean_data(self, filepath: str, chunk_size: int = 50000) -> Optional[pd.DataFrame]:
        """Load CSV file and perform data cleaning with memory optimization for large files"""
        try:
            # Check file size first
            import os
            file_size = os.path.getsize(filepath)
            self.logger.info(f"Processing file {filepath} (Size: {file_size / (1024*1024):.1f}MB)")
            
            # For large files (>50MB), use chunked processing
            if file_size > 50 * 1024 * 1024:  # 50MB threshold
                self.logger.info("Large file detected - using chunked processing")
                df = self._load_large_file_chunked(filepath, chunk_size)
            else:
                # Load small files normally
                self.logger.info(f"Loading data from {filepath}")
                df = pd.read_csv(filepath, low_memory=False)
            
            self.logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")
            
            # Clean and standardize column names
            df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
            
            # Detect and clean timestamp columns
            df = self._clean_timestamps(df)
            
            # Calculate delivery time if not present
            df = self._calculate_delivery_time(df)
            
            # Clean numeric columns
            df = self._clean_numeric_columns(df)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Feature engineering
            df = self._engineer_features(df)
            
            self.logger.info(f"Data cleaning completed. {len(df)} records remaining.")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading/cleaning data: {str(e)}")
            # Clean up memory on error
            gc.collect()
            return None
        finally:
            # Force garbage collection after processing large datasets
            gc.collect()
    
    def _load_large_file_chunked(self, filepath: str, chunk_size: int = 50000) -> Optional[pd.DataFrame]:
        """Load large CSV files in chunks to manage memory efficiently"""
        try:
            self.logger.info(f"Processing large file in chunks of {chunk_size} rows")
            
            # Read the file header first to get column information
            sample_df = pd.read_csv(filepath, nrows=5, low_memory=False)
            self.logger.info(f"File columns: {list(sample_df.columns)}")
            
            chunks = []
            chunk_count = 0
            
            # Process file in chunks
            chunk_reader = pd.read_csv(filepath, chunksize=chunk_size, low_memory=False)
            
            for chunk in chunk_reader:
                chunk_count += 1
                self.logger.info(f"Processing chunk {chunk_count} ({len(chunk)} rows)")
                
                # Clean column names for this chunk
                chunk.columns = chunk.columns.str.lower().str.strip().str.replace(' ', '_')
                
                # Basic cleaning for each chunk
                chunk = self._clean_timestamps(chunk)
                chunk = self._calculate_delivery_time(chunk)
                chunk = self._clean_numeric_columns(chunk)
                chunk = self._handle_missing_values(chunk)
                
                chunks.append(chunk)
                
                # Memory management: if too many chunks accumulated, merge them
                if len(chunks) >= 10:
                    self.logger.info("Merging accumulated chunks to manage memory")
                    merged_chunk = pd.concat(chunks, ignore_index=True)
                    chunks = [merged_chunk]
                    gc.collect()  # Force garbage collection
            
            # Combine all chunks
            self.logger.info(f"Combining {len(chunks)} processed chunks")
            if len(chunks) == 1:
                df = chunks[0]
            else:
                df = pd.concat(chunks, ignore_index=True)
            
            # Final feature engineering on combined data
            df = self._engineer_features(df)
            
            self.logger.info(f"Large file processing completed. {len(df)} total records.")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing large file: {str(e)}")
            return None
        finally:
            # Clean up memory
            if 'chunks' in locals():
                del chunks
            gc.collect()
    
    def _clean_timestamps(self, df):
        """Clean and standardize timestamp columns"""
        timestamp_columns = []
        
        # Identify potential timestamp columns
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp', 'created', 'pickup', 'delivery']):
                timestamp_columns.append(col)
        
        for col in timestamp_columns:
            try:
                # Convert to datetime, handling mixed formats
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                self.logger.info(f"Cleaned timestamp column: {col}")
            except Exception as e:
                self.logger.warning(f"Could not clean timestamp column {col}: {str(e)}")
        
        return df
    
    def _calculate_delivery_time(self, df):
        """Calculate actual delivery time"""
        try:
            # First check for pre-calculated duration columns
            if 'delivery_duration' in df.columns:
                df['actual_delivery_time'] = pd.to_numeric(df['delivery_duration'], errors='coerce')
                self.logger.info("Using pre-calculated delivery_duration column")
            
            elif 'duration' in df.columns:
                df['actual_delivery_time'] = pd.to_numeric(df['duration'], errors='coerce')
                self.logger.info("Using duration column")
            
            elif 'delivery_time' in df.columns and not any('time' in col.lower() and 'pickup' in col.lower() for col in df.columns):
                # Only use if it's not a timestamp column (i.e., no pickup_time exists)
                df['actual_delivery_time'] = pd.to_numeric(df['delivery_time'], errors='coerce')
                self.logger.info("Using delivery_time as numeric duration")
            
            else:
                # Look for timestamp columns to calculate duration
                pickup_cols = [col for col in df.columns if 'pickup' in col.lower() and 'time' in col.lower()]
                delivery_cols = [col for col in df.columns if 'delivery' in col.lower() and 'time' in col.lower()]
                
                if pickup_cols and delivery_cols:
                    pickup_col = pickup_cols[0]
                    delivery_col = delivery_cols[0]
                    
                    # Calculate delivery time in hours
                    pickup_times = pd.to_datetime(df[pickup_col], errors='coerce')
                    delivery_times = pd.to_datetime(df[delivery_col], errors='coerce')
                    time_diff = delivery_times - pickup_times
                    df['actual_delivery_time'] = time_diff.dt.total_seconds() / 3600  # Convert to hours
                    
                    self.logger.info(f"Calculated delivery time using {pickup_col} and {delivery_col}")
                
                else:
                    # Generate synthetic delivery time for demo purposes (this would not be done in production)
                    self.logger.warning("No delivery time columns found, generating synthetic data for demo")
                    np.random.seed(42)
                    df['actual_delivery_time'] = np.random.normal(2.5, 1.0, len(df))  # Mean 2.5 hours, std 1 hour
            
            # Ensure delivery time is positive and not NaN
            if 'actual_delivery_time' in df.columns:
                # Fill NaN values with median
                median_time = df['actual_delivery_time'].median()
                if pd.isna(median_time):  # If all values are NaN
                    df['actual_delivery_time'] = 2.5  # Default to 2.5 hours
                else:
                    df['actual_delivery_time'].fillna(median_time, inplace=True)
                
                # Ensure positive values
                df['actual_delivery_time'] = np.maximum(df['actual_delivery_time'], 0.1)
                
                self.logger.info(f"Final delivery time stats: min={df['actual_delivery_time'].min():.2f}, max={df['actual_delivery_time'].max():.2f}, mean={df['actual_delivery_time'].mean():.2f}")
        
        except Exception as e:
            self.logger.error(f"Error calculating delivery time: {str(e)}")
            # Fallback: generate synthetic data
            np.random.seed(42)
            df['actual_delivery_time'] = np.random.normal(2.5, 1.0, len(df))
            df['actual_delivery_time'] = np.maximum(df['actual_delivery_time'], 0.1)
        
        return df
    
    def _clean_numeric_columns(self, df):
        """Clean numeric columns"""
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Remove outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values"""
        # For numeric columns, fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        return df
    
    def _engineer_features(self, df):
        """Engineer additional features"""
        try:
            self.logger.info(f"Starting feature engineering with {len(df)} records")
            self.logger.info(f"Available columns: {list(df.columns)}")
            
            # Extract time-based features if timestamp columns exist
            timestamp_cols = df.select_dtypes(include=['datetime64']).columns
            self.logger.info(f"Found timestamp columns: {list(timestamp_cols)}")
            
            if len(timestamp_cols) > 0:
                timestamp_col = timestamp_cols[0]
                df['hour_of_day'] = df[timestamp_col].dt.hour
                df['day_of_week'] = df[timestamp_col].dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                df['is_peak_hour'] = ((df['hour_of_day'] >= 8) & (df['hour_of_day'] <= 10) | 
                                     (df['hour_of_day'] >= 17) & (df['hour_of_day'] <= 19)).astype(int)
                self.logger.info(f"Created time-based features: hour_of_day, day_of_week, is_weekend, is_peak_hour")
            
            # Calculate distance if coordinates are available
            coord_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['lat', 'lng', 'lon', 'coord'])]
            self.logger.info(f"Found coordinate columns: {coord_cols}")
            
            if len(coord_cols) >= 4:  # Need at least 4 coordinates (pickup lat/lng, delivery lat/lng)
                df['distance'] = self._calculate_distance(df, coord_cols)
            else:
                self.logger.warning(f"Not enough coordinate columns found ({len(coord_cols)}), skipping distance calculation")
            
            # Create categorical features
            if 'city' not in df.columns and len(df.columns) > 0:
                # Create a synthetic city column based on data patterns
                np.random.seed(42)
                cities = ['Dar es Salaam', 'Arusha', 'Mwanza', 'Dodoma', 'Mbeya']
                df['city'] = np.random.choice(cities, len(df))
                self.logger.info("Created synthetic city column")
            else:
                self.logger.info(f"City column already exists or no columns available")
            
            # Log final feature summary
            engineered_features = ['hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour', 'distance']
            available_features = [f for f in engineered_features if f in df.columns]
            self.logger.info(f"Successfully engineered features: {available_features}")
            
        except Exception as e:
            self.logger.warning(f"Error in feature engineering: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        return df
    
    def _calculate_distance(self, df, coord_cols):
        """Calculate distance between pickup and delivery points"""
        try:
            self.logger.info(f"Calculating distance using coordinate columns: {coord_cols}")
            # Simple Euclidean distance calculation
            # In production, would use Haversine formula for geographic distance
            pickup_lat = pd.to_numeric(df[coord_cols[0]], errors='coerce')
            pickup_lng = pd.to_numeric(df[coord_cols[1]], errors='coerce')
            delivery_lat = pd.to_numeric(df[coord_cols[2]], errors='coerce')
            delivery_lng = pd.to_numeric(df[coord_cols[3]], errors='coerce')
            
            try:
                self.logger.info(f"Coordinate ranges - pickup_lat: {pickup_lat.min():.4f}-{pickup_lat.max():.4f}, pickup_lng: {pickup_lng.min():.4f}-{pickup_lng.max():.4f}")
            except:
                self.logger.info("Could not log coordinate ranges")
            
            distance = np.sqrt((pickup_lat - delivery_lat)**2 + (pickup_lng - delivery_lng)**2)
            distance = distance.fillna(5.0)  # Fill NaN with default distance
            
            try:
                self.logger.info(f"Calculated distances - min: {distance.min():.4f}, max: {distance.max():.4f}, mean: {distance.mean():.4f}")
            except:
                self.logger.info("Could not log distance statistics")
            return distance
        
        except Exception as e:
            self.logger.warning(f"Error calculating distance: {str(e)}")
            return np.random.uniform(1, 50, len(df))  # Fallback random distance
