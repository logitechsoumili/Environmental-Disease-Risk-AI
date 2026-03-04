from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import os
import io
import uuid
from model_utils import predict_environment
from rag_utils import generate_health_advisory, answer_followup_question
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import warnings
import logging
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.abspath(os.path.join(BASE_DIR, "..", "uploads"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["LAST_ENV_CLASS"] = None

def cleanup_old_uploads(folder, max_age_seconds=7200):
    now = time.time()

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)

            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                except:
                    pass

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/report")
def report_page():
    return render_template("report.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    cleanup_old_uploads(app.config["UPLOAD_FOLDER"])
    
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)

    prediction, confidence = predict_environment(filepath)
    advisory = generate_health_advisory(prediction)

    return jsonify({
        "image": filename,
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "diseases": advisory.get("diseases", []),
        "preventive_measures": advisory.get("preventive_measures", []),
        "health_guidelines": advisory.get("health_guidelines", []),
        "rag_answer": advisory.get("rag_answer", "")
    })


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    environment_class = payload.get("environment_class", "").strip()

    if not question:
        return jsonify({"error": "Question is required."}), 400
    if not environment_class:
        return jsonify({"error": "Analyze an image first to establish context."}), 400

    answer = answer_followup_question(environment_class, question)
    return jsonify({"answer": answer})


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/download_report", methods=["POST"])
def download_report():
    data = request.get_json(silent=True) or {}
    image_filename = os.path.basename((data.get("image") or "").strip())
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename) if image_filename else None
    pdf_buffer = io.BytesIO()

    styles = getSampleStyleSheet()
    title_style = styles['Title']
    section_style = styles['Heading2']
    text_style = styles['BodyText']
    elements = []

    elements.append(Paragraph("Environmental AI Healthcare System", title_style))
    elements.append(Paragraph("Environmental Health Assessment Report", section_style))
    elements.append(Spacer(1, 20))

    if image_path and os.path.exists(image_path):
        try:
            image_reader = ImageReader(image_path)
            img_width, img_height = image_reader.getSize()
            max_width = 6.2 * inch
            max_height = 3.4 * inch
            scale = min(max_width / img_width, max_height / img_height, 1)
            report_image = RLImage(
                image_path,
                width=img_width * scale,
                height=img_height * scale
            )
            elements.append(report_image)
            elements.append(Spacer(1, 14))
        except Exception:
            elements.append(Paragraph("Uploaded image could not be embedded.", styles['Normal']))
            elements.append(Spacer(1, 10))

    prediction = data.get("prediction", "N/A")
    confidence = data.get("confidence", "N/A")
    diseases = data.get("diseases", []) or []
    preventive_measures = data.get("preventive_measures", []) or []
    health_guidelines = data.get("health_guidelines", []) or []

    elements.append(Paragraph(f"<b>Detected Environment:</b> {prediction}", text_style))
    elements.append(Paragraph(f"<b>Model Confidence:</b> {confidence}%", text_style))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Health Risks", section_style))
    for item in diseases:
        elements.append(Paragraph(f"• {item}", text_style))

    elements.append(Paragraph("Preventive Measures", section_style))
    for item in preventive_measures:
        elements.append(Paragraph(f"• {item}", text_style))

    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Health Guidelines", section_style))
    for item in health_guidelines:
        elements.append(Paragraph(f"• {item}", text_style))

    if not elements:
        elements.append(Paragraph("Environmental Health Report", styles['Title']))

    elements.append(Spacer(1, 25))
    elements.append(
        Paragraph(
            "This report was automatically generated by the Environmental AI Healthcare System "
            "using CNN-based environmental detection and Retrieval-Augmented Generation.",
            styles['Italic']
        )
    )

    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    doc.build(elements)
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="environmental_health_report.pdf"
    )


if __name__ == "__main__":
    app.run(debug=False)