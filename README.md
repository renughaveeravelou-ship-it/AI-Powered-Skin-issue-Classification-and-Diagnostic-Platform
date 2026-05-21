# AI-Based Skin Disease Prediction System

This repository contains a Flask-based deep learning application for automated skin disease classification using TensorFlow and CNN models. The system performs image-based prediction, Grad-CAM explainability visualization, lesion segmentation using OpenCV, and generates clinical-style PDF reports.
---

## 📌 Features

- Skin disease prediction using CNN model
- Grad-CAM heatmap visualization
- Lesion segmentation using OpenCV
- User authentication system
- AI skincare chatbot
- PDF medical report generation
- Dashboard with scan history
- Multiple AI engines support
  - Local CNN
  - Ensemble Model
  - Gemini API
  - OpenAI API

---

## Technologies Used

### Frontend
- HTML
- CSS
- JavaScript
- Bootstrap

### Backend
- Python
- Flask

### AI / Machine Learning
- TensorFlow
- Keras
- OpenCV
- NumPy

### Database
- SQLite

### APIs
- Gemini API
- OpenAI API

---

##Installation

###1️ Clone Repository
-git clone https://github.com/your-username/DermShield-AI.git
-cd DermShield-AI

2️ Create Virtual Environment
python -m venv venv

3️ Activate Virtual Environment
Windows
venv\Scripts\activate
Linux/Mac
source venv/bin/activate

4️ Install Dependencies
-pip install -r requirements.txt

Run the Project
python app.py

Open browser:
http://127.0.0.1:5000

3 Supported Skin Diseases
Acne
Melanoma
Peeling Skin
Ringworm
Vitiligo

AI Features
Grad-CAM Visualization
Highlights infected regions detected by the CNN model.

Lesion Segmentation
Uses OpenCV image processing for lesion boundary detection.

AI Chatbot
Provides skincare suggestions and guidance.

PDF Report Generation
The system generates:
Diagnosis report
Confidence scores
Heatmap analysis
Clinical recommendations

Authentication Features
User Registration
Login System
Session Management
Scan History Dashboard

Future Improvements
Mobile App Integration
More Skin Disease Classes
Cloud Deployment
Real-time Camera Detection
Doctor Appointment Integration

Author
-Renugha v

License
-This project is for educational and research purposes only.

---

## 📂 Project Structure

```bash
DermShield-AI/
│
├── app.py                     # Main Flask application
├── database.py                # Database operations
├── skindisease.h5             # Trained CNN model
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── uploads/                   # Uploaded images & reports
│   ├── heatmap_images/
│   ├── segmentation_images/
│   └── reports/
│
├── frontend/
│   └── dist/
│       ├── index.html
│       ├── assets/
│       ├── css/
│       └── js/
│
├── templates/
│   └── base.html
│
├── static/
│   ├── sw.js
│   ├── css/
│   ├── js/
│   └── images/
│
└── database/
    └── users.db

