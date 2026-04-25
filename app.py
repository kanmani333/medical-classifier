from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3, os, joblib, datetime, json
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np

# ── App config ────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'medicalclassifier2024secretkey'

# ── Create required folders on startup ───────────────────────
os.makedirs('uploads', exist_ok=True)
os.makedirs('instance', exist_ok=True)

app.config['UPLOAD_FOLDER']   = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED = {'pdf', 'png', 'jpg', 'jpeg'}

# ── Login manager ─────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ── Load ML model ─────────────────────────────────────────────
pipeline = joblib.load('model/svm_pipeline.pkl')
labels   = joblib.load('model/labels.pkl')

# ── Category info ─────────────────────────────────────────────
CATEGORY_INFO = {
    "Surgery": {
        "icon": "🔪",
        "color": "#FF6B6B",
        "bg": "#FFF0F0",
        "desc": "Report relates to surgical procedures and operations.",
        "keywords": ["incision", "suture", "laparoscopic", "anesthesia", "procedure",
                     "operation", "surgical", "excision", "resection", "anastomosis"]
    },
    "Radiology": {
        "icon": "🩻",
        "color": "#4ECDC4",
        "bg": "#F0FFFE",
        "desc": "Report relates to medical imaging — MRI, CT scan, X-ray.",
        "keywords": ["mri", "ct scan", "x-ray", "imaging", "contrast", "scan",
                     "radiograph", "ultrasound", "findings", "signal"]
    },
    "Neurology": {
        "icon": "🧠",
        "color": "#45B7D1",
        "bg": "#F0F8FF",
        "desc": "Report relates to brain and nervous system conditions.",
        "keywords": ["seizure", "eeg", "cranial", "cerebral", "nerve", "brain",
                     "neurological", "cortex", "spinal", "neurology"]
    }
}

# ── Database ──────────────────────────────────────────────────
def get_db():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'database.db')
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('instance', exist_ok=True)
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            extracted_text TEXT,
            category TEXT,
            confidence REAL,
            keywords TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        db.commit()

# ── User class ────────────────────────────────────────────────
class User(UserMixin):
    def __init__(self, id, name, email):
        self.id    = id
        self.name  = name
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    db   = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if user:
        return User(user['id'], user['name'], user['email'])
    return None

# ── Helpers ───────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

def extract_text(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        doc  = fitz.open(filepath)
        text = ' '.join(page.get_text() for page in doc)
        return text.strip()
    else:
        img  = Image.open(filepath)
        text = pytesseract.image_to_string(img)
        return text.strip()

def get_confidence(text):
    decision = pipeline.named_steps['svm'].decision_function(
        pipeline.named_steps['tfidf'].transform([text])
    )
    scores     = decision[0]
    exp_scores = np.exp(scores - np.max(scores))
    softmax    = exp_scores / exp_scores.sum()
    confidence = float(np.max(softmax)) * 100
    return round(confidence, 1)

def highlight_keywords(text, category):
    keywords = CATEGORY_INFO.get(category, {}).get('keywords', [])
    found    = []
    text_low = text.lower()
    for kw in keywords:
        if kw in text_low:
            found.append(kw)
    return found

# ── Routes ────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name     = request.form['name'].strip()
        email    = request.form['email'].strip()
        password = request.form['password']
        db       = get_db()
        existing = db.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
        if existing:
            flash('Email already registered. Please login.', 'error')
            return redirect(url_for('register'))
        hashed = generate_password_hash(password)
        db.execute('INSERT INTO users (name, email, password, created_at) VALUES (?, ?, ?, ?)',
                   (name, email, hashed, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        db.commit()
        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form['email'].strip()
        password = request.form['password']
        db       = get_db()
        user     = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user and check_password_hash(user['password'], password):
            login_user(User(user['id'], user['name'], user['email']))
            return redirect(url_for('upload'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(url_for('upload'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('upload'))
        if not allowed_file(file.filename):
            flash('Only PDF, PNG, JPG files allowed.', 'error')
            return redirect(url_for('upload'))

        # Save file
        filename = secure_filename(file.filename)
        upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        # Extract text
        try:
            text = extract_text(filepath)
        except Exception as e:
            flash(f'Could not extract text: {str(e)}', 'error')
            return redirect(url_for('upload'))

        if not text or len(text.strip()) < 10:
            flash('Could not extract text from file. Try a clearer image or PDF.', 'error')
            return redirect(url_for('upload'))

        # Classify
        category   = pipeline.predict([text])[0]
        confidence = get_confidence(text)
        keywords   = highlight_keywords(text, category)

        # Save to DB
        db = get_db()
        db.execute('''INSERT INTO reports (user_id, filename, extracted_text, category, confidence, keywords, created_at)
                      VALUES (?, ?, ?, ?, ?, ?, ?)''',
                   (current_user.id, filename, text[:2000], category, confidence,
                    json.dumps(keywords), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        db.commit()
        report_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]

        return redirect(url_for('result', report_id=report_id))

    return render_template('upload.html')

@app.route('/result/<int:report_id>')
@login_required
def result(report_id):
    db     = get_db()
    report = db.execute('SELECT * FROM reports WHERE id = ? AND user_id = ?',
                        (report_id, current_user.id)).fetchone()
    if not report:
        flash('Report not found.', 'error')
        return redirect(url_for('upload'))
    info     = CATEGORY_INFO.get(report['category'], {})
    keywords = json.loads(report['keywords']) if report['keywords'] else []
    return render_template('result.html', report=report, info=info, keywords=keywords)

@app.route('/history')
@login_required
def history():
    db      = get_db()
    reports = db.execute('SELECT * FROM reports WHERE user_id = ? ORDER BY created_at DESC',
                         (current_user.id,)).fetchall()
    return render_template('history.html', reports=reports)

@app.route('/delete/<int:report_id>')
@login_required
def delete_report(report_id):
    db = get_db()
    db.execute('DELETE FROM reports WHERE id = ? AND user_id = ?', (report_id, current_user.id))
    db.commit()
    flash('Report deleted.', 'success')
    return redirect(url_for('history'))

# ── Run ───────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('instance', exist_ok=True)
    init_db()
    app.run(debug=True)

# ── Init DB on import (for gunicorn) ─────────────────────────
init_db()