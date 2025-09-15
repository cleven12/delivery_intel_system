
---

# Delivery Intelligence System

## Team Members

* Faustine Emmanuel
* Cleven Godson

## Problem

Delivery operations can often be unpredictable, leading to delays, inefficiencies, or even fraud. Understanding the factors affecting delivery times and identifying anomalies can significantly improve decision-making, reduce costs, and enhance customer satisfaction. However, tracking and analyzing such data manually can be time-consuming and prone to errors.

## Solution Overview

The **Delivery Intelligence System** is a smart solution designed to streamline delivery operations using AI and data analytics. Built as a **Flask web application**, it helps delivery companies predict delivery times and detect anomalies in real-time. This system processes CSV datasets containing delivery information, providing actionable insights through a user-friendly interactive dashboard.

The key features include:

* **Prediction of delivery times** using machine learning.
* **Anomaly detection** to identify unusual delivery patterns that could indicate fraud or operational issues.
* **Interactive visualizations** to help users easily understand the data and make informed decisions.

## Instructions to Run the Project

To run the **Delivery Intelligence System** locally, follow these simple steps:

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/cleven12/delivery_intel_system.git
cd delivery_intel_system
````

### 2. Install Dependencies

Make sure you have **Python 3.x** installed. Then, install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the Flask Application

Start the Flask application by running:

```bash
cd src
python app.py
```

This will start a local web server. You can access the application by visiting `http://127.0.0.1:5000/` in your browser.

### 4. Use the Application

Once the app is running, you can:

* **Upload CSV files**: Click the "Upload File" button on the homepage to load your delivery dataset.
* **View the dashboard**: The dashboard will display predictions, anomalies, and analytics for your data.

## Special Requirements

* **Python Version**: Python 3.6 or higher.
* **Libraries**:

  * Flask
  * Pandas
  * NumPy
  * Scikit-learn
  * Chart.js (for visualizations)
  * Bootstrap (for the UI)

You can install all the dependencies by running:

```bash
pip install -r requirements.txt
```

* **File Handling**: The system handles CSV file uploads up to 200MB. Ensure your dataset is in CSV format and has proper columns (e.g., delivery times, city names, timestamps).

## Deployment

You can access the live version of the system at the following URL:
**[Kilitech Duo - Delivery Intelligence System](https://anna2tx.pythonanywhere.com)**

The code for this project is available on GitHub:
**[GitHub Repository](https://github.com/cleven12/delivery_intel_system)**

## System Architecture

### Frontend Architecture

* **Responsive UI**: Built using Jinja2 templates and **Bootstrap**, ensuring a mobile-friendly, sleek design with a dark theme.
* **Interactive Charts**: Data visualizations powered by **Chart.js**, including real-time analytics for delivery predictions and anomaly detection.
* **Single-Page Application Flow**: Upload → Processing → Dashboard, with persistent session data.

### Backend Architecture

* **Flask Web Framework**: Simple and lightweight Python server that handles all web requests and serves the dashboard.

* **Modular Design**:

  * `app.py`: Manages routing and requests.
  * `data_processor.py`: Handles data cleaning and preprocessing.
  * `ml_models.py`: Contains the machine learning models for predictions and anomaly detection.

* **Session Data**: Uses in-memory session storage for temporary data during the demo.

### Machine Learning Architecture

* **Prediction Model**: A custom linear regression model to predict delivery times using **NumPy**.
* **Anomaly Detection**: Uses statistical methods (isolation techniques) to detect unusual delivery patterns.
* **Training Pipeline**: Data cleaning → Feature extraction → Model training → Evaluation.

### Data Processing Pipeline

* **CSV Ingestion**: Uses **Pandas** to load and process delivery datasets.
* **Data Cleaning**: Standardizes timestamps, handles missing data, and detects outliers.
* **Feature Engineering**: Extracts features like delivery durations and normalizes numerical columns.
* **Validation**: Ensures data quality with error handling during the process.

## External Dependencies

### Core Web Framework

* **Flask**: The backbone of the web application.
* **Werkzeug**: Provides utilities like secure filename handling and middleware support.

### Data Science Stack

* **Pandas**: Used for CSV processing and data manipulation.
* **NumPy**: For numerical operations and custom machine learning models.
* **Scikit-learn**: Provides metrics for model evaluation (e.g., MAE, MSE, R²).

### Frontend Libraries

* **Bootstrap**: A front-end framework for responsive design and mobile-friendly UI.
* **Chart.js**: For creating interactive visualizations of data like predicted vs actual delivery times.
* **Font Awesome**: Enhances the UI with icons.

### Development Tools

* **Python Logging**: Built-in logging for tracking errors and debugging.
* **Werkzeug ProxyFix**: Middleware for header handling during deployment.

### File System Dependencies

* **OS Module**: For environment variable management and directory handling.
* **Uploads Directory**: Used for storing CSV files temporarily during processing.

---
