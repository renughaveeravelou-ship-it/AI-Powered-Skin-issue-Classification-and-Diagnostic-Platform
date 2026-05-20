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

## 🛠️ Technologies Used

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

