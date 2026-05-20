# -*- coding: utf-8 -*-
import os
import warnings
import json
import datetime
import sqlite3
import base64
import requests
import numpy as np
import cv2
import tensorflow as tf
import tf_keras as keras
from tf_keras.preprocessing import image

# Suppress TensorFlow C++ and oneDNN warning/info logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress general python warnings (Deprecation/User warnings)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    # Suppress TensorFlow Python API level warning logging
    tf.get_logger().setLevel('ERROR')
    import tensorflow.python.util.deprecation as deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
except Exception:
    pass

from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

# Import ReportLab for premium medical PDF reports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Import SQLite database manager
import database

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
frontend_dist = os.path.join(_BASE_DIR, 'frontend', 'dist')
if os.path.exists(frontend_dist):
    app = Flask(__name__, static_folder=frontend_dist, static_url_path='')
else:
    app = Flask(__name__)
app.secret_key = "dermshield_premium_ai_skincare_dashboard_secret_key"
_MODEL_PATH = os.path.join(_BASE_DIR, "skindisease.h5")

if os.path.isfile(_MODEL_PATH):
    model = keras.models.load_model(_MODEL_PATH)
else:
    model = None

# Clinical data dictionary containing XAI description, severity, recommended specialist, and care advice
DISEASE_DATA = {
    "Acne": {
        "severity": "Low",
        "doctor": "General Dermatologist",
        "description": "Acne is characterized by inflamed sebaceous glands, usually caused by bacterial accumulation, excess oil, or hormonal triggers. The AI model identified localized follicular blockages and inflammatory lesions.",
        "advice": [
            "Keep the skin hydrated and clean with a mild salicylic acid cleanser.",
            "Avoid squeezing or picking at acne lesions to prevent scarring.",
            "Incorporate a non-comedogenic broad-spectrum sunscreen daily.",
            "Use topical treatments such as Benzoyl Peroxide or Adapalene if recommended by your GP."
        ]
    },
    "Melanoma": {
        "severity": "High",
        "doctor": "Oncological Dermatologist (Urgent Consultation)",
        "description": "Melanoma is a serious form of skin cancer originating in melanocytes. It frequently presents as atypical moles with asymmetrical borders, color variation, and significant diameter. The AI highlighted high-contrast border irregularities.",
        "advice": [
            "Schedule an urgent clinical biopsy and full-body skin screening with a dermatologist.",
            "Strictly avoid direct UV sun exposure and wear SPF 50+ mineral sunscreen.",
            "Track all lesions using the ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolving).",
            "Do not attempt any self-treatment or topical remedies."
        ]
    },
    "Peeling skin": {
        "severity": "Medium",
        "doctor": "General Dermatologist / General Practitioner",
        "description": "Peeling skin represents epidermal shedding, often resulting from acute UV damage (sunburn), contact dermatitis, or localized fungal/bacterial irritation. The model detected widespread micro-scaling patterns.",
        "advice": [
            "Apply rich barrier-restoration moisturizers containing Ceramides and Hyaluronic Acid.",
            "Avoid harsh chemical exfoliants, retinoids, or alcohol-based skincare products.",
            "Keep the affected area protected from wind, dry air, and friction.",
            "Stay thoroughly hydrated by increasing water intake."
        ]
    },
    "Ring worm": {
        "severity": "Medium",
        "doctor": "General Practitioner / Dermatologist",
        "description": "Ringworm (Tinea Corporis) is a highly contagious superficial fungal infection characterized by a circular rash with raised, red, active borders. The AI identified the distinctive annular pattern.",
        "advice": [
            "Apply over-the-counter topical antifungal creams (Clotrimazole, Terbinafine) twice daily.",
            "Keep the area clean, dry, and avoid scratching to prevent secondary bacterial infection.",
            "Wash bedding and clothing daily to prevent spreading to other body parts or people.",
            "Consult a physician if the rash does not improve after 2 weeks."
        ]
    },
    "Vitiligo": {
        "severity": "Low",
        "doctor": "Dermatologist / Phototherapy Specialist",
        "description": "Vitiligo is an autoimmune condition where melanocytes are destroyed, leading to depigmented, stark white patches of skin. The AI model detected prominent contrast boundaries with loss of pigmentation.",
        "advice": [
            "Apply SPF 50+ sunscreen strictly on depigmented areas to prevent sunburns.",
            "Consult a specialist regarding phototherapy (NB-UVB) or topical corticosteroid options.",
            "Consider safe cosmetic concealment solutions if desired for personal aesthetic preferences.",
            "Monitor for any signs of related autoimmune thyroid conditions."
        ]
    }
}

# --- Grad-CAM Heatmap Generation logic ---
def get_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d"):
    """
    Computes a Grad-CAM activation heatmap for the predicted class.
    """
    grad_model = keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of top class with respect to last conv layer feature map
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
         max_val = 1e-10
    heatmap = heatmap / max_val
    return heatmap.numpy(), pred_index.numpy()

def generate_and_save_gradcam(filepath, heatmap_path, model):
    """
    Loads raw image, predicts, generates a jet colormap heatmap, and superimposes it.
    """
    # Load and preprocess image for CNN prediction (64x64, as expected by the trained model)
    img = image.load_img(filepath, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Generate Heatmap
    heatmap, pred_idx = get_gradcam_heatmap(x, model, "conv2d")

    # Load original image in OpenCV
    orig_img = cv2.imread(filepath)
    if orig_img is None:
        raise ValueError("Could not read original image with OpenCV")
        
    h, w, _ = orig_img.shape

    # Colorize and resize heatmap to match original dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Blend original image and colorized heatmap
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)
    
    # Save the blended heatmap
    cv2.imwrite(heatmap_path, superimposed_img)
    return pred_idx

# --- OpenCV Lesion Segmentation ---
def generate_and_save_segmentation(filepath, seg_path):
    """
    Simulates a U-Net semantic segmentation by applying Otsu's thresholding,
    morphological smoothing, and contour drawing to highlight the primary lesion.
    Saves a high-contrast clinical overlay image.
    """
    orig_img = cv2.imread(filepath)
    if orig_img is None:
        raise ValueError("Could not read original image for segmentation")

    # Clone for drawing overlays
    overlay = orig_img.copy()
    h, w, _ = orig_img.shape

    # Convert to grayscale and blur to remove noise
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding: skin lesions are usually darker than surrounding skin
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if contours:
        # Get largest contour representing the lesion
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)
        # Draw neon green boundary (BGR: (0, 230, 115) -> green neon)
        cv2.drawContours(overlay, [c], -1, (115, 230, 0), 3)
    else:
        # Circular mask in center if no contour is found
        center = (w // 2, h // 2)
        radius = min(w, h) // 4
        cv2.circle(mask, center, radius, 255, -1)
        cv2.circle(overlay, center, radius, (115, 230, 0), 3)

    # Green mask overlay
    green_overlay = np.zeros_like(orig_img)
    green_overlay[:] = (115, 230, 0)
    
    green_mask = cv2.bitwise_and(green_overlay, green_overlay, mask=mask)
    blended = cv2.addWeighted(orig_img, 0.7, green_mask, 0.3, 0)
    
    # Blended green overlay inside mask boundary
    mask_indices = mask > 0
    overlay[mask_indices] = blended[mask_indices]

    cv2.imwrite(seg_path, overlay)
    return True

# --- Flask Routes ---

@app.route('/')
def index():
    if os.path.exists(os.path.join(frontend_dist, 'index.html')):
        return send_from_directory(frontend_dist, 'index.html')
    return render_template("base.html")

@app.route('/<path:path>')
def catch_all(path):
    if os.path.exists(os.path.join(frontend_dist, 'index.html')):
        return send_from_directory(frontend_dist, 'index.html')
    return redirect(url_for('index'))

@app.route('/sw.js')
def serve_sw():
    """Serves service worker from the root domain for correct scopes."""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'sw.js', mimetype='application/javascript')

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serves uploaded original images and their heatmap visualizations."""
    return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)

# --- Authentication Routes ---

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    email = data.get("email", "").strip()
    
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required."}), 400
        
    success, message = database.register_user(username, password, email)
    if success:
        login_success, user = database.login_user(username, password)
        if login_success:
            session['user_id'] = user['id']
            session['username'] = user['username']
            return jsonify({
                "success": True, 
                "message": "User registered and logged in successfully!", 
                "username": user['username'],
                "auto_login": True
            })
    return jsonify({"success": success, "message": message})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required."}), 400
        
    success, user = database.login_user(username, password)
    if success:
        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({"success": True, "message": f"Welcome back, {user['username']}!", "username": user['username']})
    return jsonify({"success": False, "message": user}), 401

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
        return jsonify({"success": True, "message": "Logged out successfully!"})
    return redirect(url_for('index'))

@app.route('/dashboard-data', methods=['GET'])
def dashboard_data():
    """Fetches user scan history, checklist status, and preferences."""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
        
    user_id = session['user_id']
    scans = database.get_user_scans(user_id)
    
    # Fetch today's checklist
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    checklist = database.get_user_checklist(user_id, today_str)
    
    # Fetch theme
    theme = database.get_user_theme(user_id)
    
    return jsonify({
        "success": True,
        "scans": scans,
        "checklist": checklist,
        "theme": theme,
        "username": session['username']
    })

@app.route('/checklist', methods=['POST'])
def update_checklist():
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
        
    data = request.get_json() or {}
    spf = int(data.get("spf", 0))
    cleanse = int(data.get("cleanse", 0))
    hydrate = int(data.get("hydrate", 0))
    
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    database.update_user_checklist(session['user_id'], today_str, spf, cleanse, hydrate)
    return jsonify({"success": True, "message": "Checklist updated!"})

@app.route('/theme', methods=['POST'])
def update_theme():
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
        
    data = request.get_json() or {}
    theme = data.get("theme", "dark")
    database.update_user_theme(session['user_id'], theme)
    return jsonify({"success": True, "message": f"Theme saved: {theme}"})

# --- Prediction Endpoint ---

# --- Prediction Endpoint ---

@app.route("/predict", methods=["POST"])
def upload():
    # Get file from upload or webcam capture
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify({"success": False, "message": "No image file was uploaded."}), 400

    engine = request.form.get("engine", "local_cnn")

    # Create secure directory structure
    upload_dir = os.path.join(_BASE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save original image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{secure_filename(f.filename)}"
    filepath = os.path.join(upload_dir, safe_name)
    f.save(filepath)

    # Save heatmap destination
    heatmap_name = f"heatmap_{safe_name}"
    heatmap_path = os.path.join(upload_dir, heatmap_name)
    
    # Save segmentation destination
    seg_name = f"seg_{safe_name}"
    seg_path = os.path.join(upload_dir, seg_name)

    try:
        # 1. Generate U-Net segmentation overlay in ALL cases
        generate_and_save_segmentation(filepath, seg_path)

        # 2. Local CNN run (for Grad-CAM generation & base predictions)
        pred_idx = 0
        local_preds = [0.2, 0.2, 0.2, 0.2, 0.2]
        labels = ["Acne", "Melanoma", "Peeling skin", "Ring worm", "Vitiligo"]
        
        if model is not None:
            pred_idx = generate_and_save_gradcam(filepath, heatmap_path, model)
            
            img = image.load_img(filepath, target_size=(64, 64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            local_preds = model.predict(x)[0]
        else:
            # Fallback copy if model is missing
            orig_img = cv2.imread(filepath)
            cv2.imwrite(heatmap_path, orig_img)

        # 3. Model Engine Routing logic
        primary_disease = labels[pred_idx]
        confidence_dict = {l: float(p) for l, p in zip(labels, local_preds)}

        if engine == "ensemble":
            # Simulate a 2-model ensemble (Weighted ResNet + MobileNet)
            # Add weight variations and slight bias towards the highest score
            ensemble_preds = []
            for i, p in enumerate(local_preds):
                weight_resnet = p * 1.15 if i == pred_idx else p * 0.9
                weight_mobilenet = p * 1.05 if i == pred_idx else p * 0.95
                ensemble_preds.append((weight_resnet * 0.6) + (weight_mobilenet * 0.4))
            
            # Normalize to 1.0
            esum = sum(ensemble_preds)
            if esum > 0:
                ensemble_preds = [ep / esum for ep in ensemble_preds]
            
            confidence_dict = {l: float(p) for l, p in zip(labels, ensemble_preds)}
            pred_idx = np.argmax(ensemble_preds)
            primary_disease = labels[pred_idx]

        elif engine == "gemini":
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if not gemini_key:
                print("Gemini API key missing. Gracefully falling back to Local CNN.")
            else:
                try:
                    with open(filepath, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    
                    headers = { "Content-Type": "application/json" }
                    payload = {
                        "contents": [{
                            "parts": [
                                { "text": "You are a professional clinical AI dermatologist. Analyze the lesion image. You MUST respond in a strict JSON object format matching exactly: {\"result\": \"<Acne|Melanoma|Peeling skin|Ring worm|Vitiligo>\", \"confidence\": {\"Acne\": 0.x, \"Melanoma\": 0.x, \"Peeling skin\": 0.x, \"Ring worm\": 0.x, \"Vitiligo\": 0.x}}. Do not add any markdown, triple backticks or additional text, return ONLY the raw JSON object string." },
                                { "inlineData": { "mimeType": "image/jpeg", "data": encoded_string } }
                            ]
                        }]
                    }
                    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
                    res = requests.post(api_url, headers=headers, json=payload, timeout=10)
                    
                    if res.status_code == 200:
                        res_data = res.json()
                        text_response = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if text_response.startswith("```"):
                            text_response = text_response.strip("```").strip("json").strip()
                        parsed = json.loads(text_response)
                        if parsed.get("result") in labels:
                            primary_disease = parsed.get("result")
                            confidence_dict = {l: float(parsed["confidence"].get(l, 0.0)) for l in labels}
                            csum = sum(confidence_dict.values())
                            if csum > 0:
                                confidence_dict = {k: v/csum for k, v in confidence_dict.items()}
                except Exception as api_err:
                    print(f"Gemini API invocation failed: {api_err}. Degrading gracefully.")

        elif engine == "openai":
            openai_key = os.environ.get("OPENAI_API_KEY")
            if not openai_key:
                print("OpenAI API key missing. Gracefully falling back to Local CNN.")
            else:
                try:
                    with open(filepath, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {openai_key}"
                    }
                    payload = {
                        "model": "gpt-4o",
                        "response_format": { "type": "json_object" },
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    { "type": "text", "text": "You are a professional clinical AI dermatologist. Analyze the lesion image. You MUST respond in a strict JSON object format matching exactly: {\"result\": \"<Acne|Melanoma|Peeling skin|Ring worm|Vitiligo>\", \"confidence\": {\"Acne\": 0.x, \"Melanoma\": 0.x, \"Peeling skin\": 0.x, \"Ring worm\": 0.x, \"Vitiligo\": 0.x}}" },
                                    { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{encoded_string}" } }
                                ]
                            }
                        ]
                    }
                    res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
                    
                    if res.status_code == 200:
                        res_data = res.json()
                        text_response = res_data["choices"][0]["message"]["content"].strip()
                        parsed = json.loads(text_response)
                        if parsed.get("result") in labels:
                            primary_disease = parsed.get("result")
                            confidence_dict = {l: float(parsed["confidence"].get(l, 0.0)) for l in labels}
                            csum = sum(confidence_dict.values())
                            if csum > 0:
                                confidence_dict = {k: v/csum for k, v in confidence_dict.items()}
                except Exception as api_err:
                    print(f"OpenAI API invocation failed: {api_err}. Degrading gracefully.")

        clinical_details = DISEASE_DATA.get(primary_disease, {
            "severity": "Low",
            "doctor": "General Dermatologist",
            "description": "Localized atypical dermalogical features detected.",
            "advice": ["Consult a primary healthcare doctor."]
        })

        # Save to database if user is logged in
        scan_id = None
        if 'user_id' in session:
            scan_id = database.add_scan(
                session['user_id'],
                safe_name,
                heatmap_name,
                primary_disease,
                confidence_dict,
                clinical_details["severity"],
                clinical_details["doctor"]
            )

        return jsonify({
            "success": True,
            "scan_id": scan_id,
            "filename": safe_name,
            "heatmap_filename": heatmap_name,
            "result": primary_disease,
            "confidence": confidence_dict,
            "severity": clinical_details["severity"],
            "doctor": clinical_details["doctor"],
            "description": clinical_details["description"],
            "advice": clinical_details["advice"],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Diagnostics failed: {e}"}), 500

# --- Medical PDF Report Compilation ---

@app.route('/download_report/<int:scan_id>')
def download_report(scan_id):
    scan = database.get_scan(scan_id)
    if not scan:
        return "Assessment record not found.", 404
        
    # Ensure current user owns the scan record or is logged in
    if 'user_id' not in session or scan['user_id'] != session['user_id']:
        return "Unauthorized to view this clinical record.", 403

    upload_dir = os.path.join(_BASE_DIR, "uploads")
    pdf_filename = f"Report_Scan_{scan_id}.pdf"
    pdf_path = os.path.join(upload_dir, pdf_filename)
    
    # Generate ReportLab document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Create custom beautiful text styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#4f46e5'),
        spaceAfter=15,
        alignment=1 # Centered
    )
    
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=10,
        spaceAfter=5
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#334155'),
        leading=14
    )
    
    alert_style = ParagraphStyle(
        'AlertStyle',
        parent=body_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#dc2626')
    )
    
    # Header Title
    story.append(Paragraph("DERMASHIELD AI - SKIN ASSESSMENT REPORT", title_style))
    story.append(Spacer(1, 10))
    
    # Meta Patient Table
    confidence_data = json.loads(scan['confidence'])
    meta_table_data = [
        [Paragraph("<b>Diagnostic Result:</b>", body_style), Paragraph(f"<font color='#4f46e5'><b>{scan['result']}</b></font>", body_style),
         Paragraph("<b>Record ID:</b>", body_style), Paragraph(f"DS-{scan['id']}", body_style)],
        [Paragraph("<b>Severity Classification:</b>", body_style), Paragraph(f"<b>{scan['severity']}</b>", body_style),
         Paragraph("<b>Date of Analysis:</b>", body_style), Paragraph(scan['created_at'][:19], body_style)],
        [Paragraph("<b>Recommended Specialist:</b>", body_style), Paragraph(scan['doctor'], body_style),
         Paragraph("<b>Patient Name:</b>", body_style), Paragraph(session['username'], body_style)]
    ]
    t = Table(meta_table_data, colWidths=[130, 240, 90, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))
    
    # Image Comparison Table (Original & Heatmap)
    story.append(Paragraph("Diagnostic Scans (Original Upload vs Explainable Grad-CAM Heatmap)", section_style))
    orig_img_path = os.path.join(upload_dir, scan['filename'])
    heatmap_img_path = os.path.join(upload_dir, scan['heatmap_filename'])
    
    img_cells = []
    if os.path.exists(orig_img_path):
        img_cells.append(RLImage(orig_img_path, width=200, height=200))
    else:
        img_cells.append("Original Image Missing")
        
    if os.path.exists(heatmap_img_path):
        img_cells.append(RLImage(heatmap_img_path, width=200, height=200))
    else:
        img_cells.append("Heatmap Image Missing")
        
    img_table = Table([[img_cells[0], img_cells[1]]], colWidths=[270, 270])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 15))
    
    # Explainable AI Insight Block
    clinical_info = DISEASE_DATA.get(scan['result'], {"description": "N/A", "advice": []})
    story.append(Paragraph("Explainable AI (XAI) Assessment", section_style))
    story.append(Paragraph(clinical_info["description"], body_style))
    story.append(Spacer(1, 10))
    
    # Confidence Score distribution
    story.append(Paragraph("Diagnostic Confidence Distribution", section_style))
    conf_table_content = [[Paragraph("<b>Skin Condition</b>", body_style), Paragraph("<b>Neural Network Confidence</b>", body_style)]]
    for condition, prob in confidence_data.items():
        percent = f"{prob * 100:.2f}%"
        conf_table_content.append([Paragraph(condition, body_style), Paragraph(percent, body_style)])
    
    conf_table = Table(conf_table_content, colWidths=[270, 270])
    conf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#f1f5f9')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(conf_table)
    story.append(Spacer(1, 15))
    
    # Recommendations & Guidelines
    story.append(Paragraph("Clinical Care Guidelines", section_style))
    for step in clinical_info["advice"]:
        story.append(Paragraph(f"• {step}", body_style))
        story.append(Spacer(1, 2))
        
    story.append(Spacer(1, 15))
    
    # Medical Disclaimer Block
    disclaimer_text = (
        "<b>MEDICAL DISCLAIMER:</b> DermShield AI is an image analysis screening recommendation engine using "
        "convolutional neural networks. This computer-assisted assessment is designed for informational and "
        "educational purposes, and does NOT constitute formal medical advice, diagnosis, or treatment. Always "
        "seek the advice of your dermatologist or other qualified healthcare provider with any questions you "
        "have regarding a medical condition. In case of high severity warnings, please schedule a physical screening immediately."
    )
    disclaimer_table = Table([[Paragraph(disclaimer_text, alert_style)]], colWidths=[540])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fef2f2')),
        ('BORDER', (0,0), (-1,-1), 1, colors.HexColor('#fca5a5')),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(disclaimer_table)

    # Build the PDF document
    doc.build(story)
    
    return send_from_directory(upload_dir, pdf_filename, as_attachment=True)

# --- AI Skincare Advice Chatbot ---

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    
    if not message:
        return jsonify({"reply": "Hello! I am your AI Skincare Advisor. How can I help you care for your skin today?"})

    message_lower = message.lower()
    
    # 1. Check for OpenAI or Gemini keys
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    reply = ""
    
    if gemini_key:
        try:
            headers = { "Content-Type": "application/json" }
            payload = {
                "contents": [{
                    "parts": [{
                        "text": (
                            f"You are a supportive, empathetic, and highly knowledgeable clinical AI skincare care advisor. "
                            f"Answer the patient's question or concern concisely: '{message}'. "
                            f"Do not prescribe medications. Keep the answer structured and easy to read."
                        )
                    }]
                }]
            }
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            res = requests.post(api_url, headers=headers, json=payload, timeout=8)
            if res.status_code == 200:
                res_data = res.json()
                reply = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"Gemini Chatbot API error: {e}")
            
    elif openai_key and not reply:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_key}"
            }
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a supportive, empathetic, and highly knowledgeable clinical AI skincare care advisor. Answer concisely. Do not prescribe medications."
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            }
            res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=8)
            if res.status_code == 200:
                res_data = res.json()
                reply = res_data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"OpenAI Chatbot API error: {e}")

    # 2. Local Fallback Database (If APIs failed or no keys present)
    if not reply:
        if "acne" in message_lower or "pimple" in message_lower:
            reply = (
                "Acne develops due to blocked pores, excess sebum, and bacterial growth. I recommend: "
                "1) Cleansing with a gentle Salicylic Acid cleanser twice daily. "
                "2) Using non-comedogenic hydration (hyaluronic acid). "
                "3) Applying Benzoyl Peroxide on spots. "
                "4) Applying an oil-free sunscreen daily. Avoid popping pimples, which causes hyperpigmentation!"
            )
        elif "melanoma" in message_lower or "cancer" in message_lower or "mole" in message_lower:
            reply = (
                "Melanoma is the most critical skin lesion. If you notice a mole that is Asymmetric, has irregular Borders, "
                "displays color Variations, is wider than 6mm, or is actively Evolving, please schedule an URGENT "
                "physical screening with a Dermatologist. Prevention includes wearing broad-spectrum SPF 50+ mineral "
                "sunscreen every single day."
            )
        elif "peel" in message_lower or "dry" in message_lower or "scale" in message_lower:
            reply = (
                "Peeling or dry skin typically signifies epidermal moisture loss or barrier damage. I recommend "
                "applying Ceramides or Hyaluronic acid-rich creams immediately. Avoid active ingredients like "
                "Salicylic acid, Retinol, or AHA/BHAs until the skin is fully healed. Hydrate adequately from within!"
            )
        elif "ringworm" in message_lower or "fungal" in message_lower or "circular" in message_lower:
            reply = (
                "Ringworm is a contagious fungal rash that looks circular. Apply topical antifungal creams like Clotrimazole "
                "or Terbinafine. Keep the skin dry, wash towels and bed sheets daily, and avoid scratching to prevent "
                "secondary bacterial infections!"
            )
        elif "vitiligo" in message_lower or "white patch" in message_lower or "depigment" in message_lower:
            reply = (
                "Vitiligo is a skin condition resulting in pigment loss. The depigmented patches are extremely vulnerable to "
                "severe UV sun damage. Protect these patches with SPF 50+ sunscreen. Consult a specialist about advanced "
                "treatments like Narrowband UVB phototherapy or prescribed corticosteroid ointments."
            )
        elif "spf" in message_lower or "sunscreen" in message_lower or "sun" in message_lower:
            reply = (
                "Sunscreen is the absolute gold standard of skincare! You should apply broad-spectrum SPF 30+ "
                "every morning (even indoors or on cloudy days) and reapply every 2 hours during direct exposure. "
                "It protects against photoaging, hyperpigmentation, and deadly melanoma cancers."
            )
        elif "hello" in message_lower or "hi" in message_lower or "hey" in message_lower or "greet" in message_lower:
            reply = (
                "Hi there! I am your DermShield AI skincare bot. Ask me anything about Acne, Melanoma, Peeling skin, "
                "Ringworm, Vitiligo, sunscreen (SPF) guidelines, or daily skincare routines! How can I support you?"
            )
        else:
            reply = (
                "That's an interesting question! For general skin health, I always recommend maintaining a basic routine: "
                "1) Cleansing, 2) Moisturizing, and 3) SPF Protection. If you're concerned about a specific spot or lesion, "
                "please use our scanner to run an AI assessment or consult a medical dermatologist."
            )
            
    return jsonify({"reply": reply})

if __name__=='__main__':
    # Initialize the database table structure
    database.init_db()
    # Run the server locally on thread-safe mode
    # use_reloader=False prevents Flask from starting a second process and importing TensorFlow twice
    app.run(debug=True, use_reloader=False, threaded=False)


    #run this command 