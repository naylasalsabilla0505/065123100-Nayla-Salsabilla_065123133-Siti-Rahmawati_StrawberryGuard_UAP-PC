from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pickle
import os
import base64
from werkzeug.utils import secure_filename
from skimage.feature import graycomatrix, graycoprops

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
IMG_SIZE = (128, 128)

# Load model
with open('model_knn.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('model_nb.pkl', 'rb') as f:
    nb = pickle.load(f)

CLASSES = {0: 'Healthy', 1: 'Leaf Spot', 2: 'Powdery Mildew'}

CLASS_INFO = {
    'Healthy': {
        'desc': 'Daun stroberi dalam kondisi sehat, tidak terdeteksi tanda-tanda penyakit.',
        'color': '#2d6a4f',
        'icon': '🌿',
        'recommendation': 'Pertahankan kondisi perawatan saat ini. Pastikan penyiraman dan pemupukan rutin.'
    },
    'Leaf Spot': {
        'desc': 'Terdeteksi bercak daun (Leaf Spot) yang disebabkan oleh jamur Mycosphaerella fragariae.',
        'color': '#b5451b',
        'icon': '🍂',
        'recommendation': 'Segera buang daun yang terinfeksi. Aplikasikan fungisida berbahan aktif captan atau myclobutanil.'
    },
    'Powdery Mildew': {
        'desc': 'Terdeteksi penyakit embun tepung (Powdery Mildew) yang disebabkan oleh jamur Podosphaera aphanis.',
        'color': '#7b6d47',
        'icon': '🌫️',
        'recommendation': 'Tingkatkan sirkulasi udara di sekitar tanaman. Aplikasikan fungisida sulfur atau kalium bikarbonat.'
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Median Filter 
def apply_median_filter(image, kernel_size=5):
    """Reduksi noise menggunakan Median Filter (kernel 5×5)."""
    return cv2.medianBlur(image, kernel_size)

# Segmentasi Warna HSV 
def segment_hsv(image):
    """Segmentasi area daun berdasarkan ruang warna HSV."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([20, 20, 20])
    upper_green = np.array([95, 255, 255])
    lower_brown = np.array([5, 30, 30])
    upper_brown = np.array([22, 255, 255])
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 60, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask_combined = cv2.bitwise_or(mask_green,
                    cv2.bitwise_or(mask_brown, mask_white))

    kernel = np.ones((5, 5), np.uint8)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)

    segmented = cv2.bitwise_and(image, image, mask=mask_combined)
    return segmented, hsv

# Ekstraksi Fitur GLCM 
def extract_glcm_features(gray_image):
    """Ekstraksi 6 fitur tekstur GLCM (Gray-Level Co-occurrence Matrix).

    Fitur yang diekstrak:
        contrast, dissimilarity, homogeneity, energy, correlation, ASM
    GLCM dihitung pada 4 sudut (0°, 45°, 90°, 135°) kemudian dirata-ratakan
    sehingga bersifat rotation-invariant.
    """
    glcm = graycomatrix(
        gray_image,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    props = ['contrast', 'dissimilarity', 'homogeneity',
             'energy', 'correlation', 'ASM']
    features = []
    for prop in props:
        val = graycoprops(glcm, prop=prop)
        features.append(float(val.mean()))  # rata-rata 4 sudut → 1 nilai/fitur

    return features  # 6 fitur GLCM

# Ekstraksi Fitur HSV + GLCM 
def extract_features(image):
    """fitur HSV (12) dan GLCM (6) → total 18 fitur.

    Pipeline:
        1. Segmentasi HSV → statistik channel H, S, V (mean, std, max, min)
        2. Konversi grayscale → GLCM → 6 properti tekstur
    """
    _, hsv = segment_hsv(image)

    # Fitur HSV — 12 fitur (4 statistik × 3 channel)
    hsv_features = []
    for i in range(3):  # H, S, V
        channel = hsv[:, :, i]
        hsv_features.append(np.mean(channel))
        hsv_features.append(np.std(channel))
        hsv_features.append(np.max(channel))
        hsv_features.append(np.min(channel))

    # Fitur GLCM — 6 fitur tekstur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm_features = extract_glcm_features(gray)

    return hsv_features + glcm_features  

def image_to_base64(img_bgr):
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def predict(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, IMG_SIZE)

    # Preprocessing
    img_filtered  = apply_median_filter(img, kernel_size=5)
    img_segmented, _ = segment_hsv(img_filtered)

    # Ekstraksi fitur HSV + GLCM
    features     = extract_features(img_filtered)
    features_arr = np.array(features).reshape(1, -1)

    # Prediksi
    pred_knn = knn.predict(features_arr)[0]
    pred_nb  = nb.predict(features_arr)[0]
    prob_knn = knn.predict_proba(features_arr)[0]
    prob_nb  = nb.predict_proba(features_arr)[0]

    knn_probs = {CLASSES[i]: round(float(prob_knn[i]) * 100, 2) for i in range(3)}
    nb_probs  = {CLASSES[i]: round(float(prob_nb[i])  * 100, 2) for i in range(3)}

    # Kembalikan juga nilai fitur GLCM untuk ditampilkan di frontend
    gray_filtered = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
    glcm_vals = extract_glcm_features(gray_filtered)
    glcm_labels = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
    glcm_display = {glcm_labels[i]: round(glcm_vals[i], 6) for i in range(6)}

    return {
        'knn': {
            'label'     : CLASSES[pred_knn],
            'confidence': round(float(prob_knn[pred_knn]) * 100, 2),
            'all_probs' : knn_probs
        },
        'nb': {
            'label'     : CLASSES[pred_nb],
            'confidence': round(float(prob_nb[pred_nb]) * 100, 2),
            'all_probs' : nb_probs
        },
        'images': {
            'original' : image_to_base64(img),
            'filtered' : image_to_base64(img_filtered),
            'segmented': image_to_base64(img_segmented)
        },
        'glcm_features': glcm_display
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dikirim'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if file and allowed_file(file.filename):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predict(filepath)
        result['class_info'] = {
            'knn': CLASS_INFO[result['knn']['label']],
            'nb' : CLASS_INFO[result['nb']['label']]
        }
        return jsonify(result)

    return jsonify({'error': 'Format file tidak didukung. Gunakan JPG atau PNG.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
