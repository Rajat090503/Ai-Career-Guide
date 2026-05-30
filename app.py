from flask import Flask, render_template, request
import os
import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from utils.resume_parser import extract_text_from_pdf
from utils.skill_extractor import extract_skills
from utils.course_recommender import recommend_courses
from utils.ats_score import calculate_ats_score
from utils.resume_suggestions import get_suggestions
from utils.roadmap_generator import generate_roadmap

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



model = load_model("deep_role_model.h5")

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ==========================
# Load Role Skills Dataset
# ==========================

skills_df = pd.read_csv("roles_skills.csv")

# ==========================
# Home Page
# ==========================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================
# Resume Analysis
# ==========================

@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files["resume"]

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    # Extract Resume Text
    text = extract_text_from_pdf(filepath)

    # Extract Skills
    skills = extract_skills(text)

    # Deep Learning Prediction
    X = vectorizer.transform([text]).toarray()

    prediction = model.predict(
        X,
        verbose=0
    )

    role = encoder.inverse_transform(
        [np.argmax(prediction)]
    )[0]

    print("Predicted Role:", role)

    # ==========================
    # Required Skills
    # ==========================

    required_skills = skills_df[
        skills_df["role"] == role
    ]["skill"].tolist()

    # ==========================
    # Missing Skills
    # ==========================

    missing_skills = []

    for skill in required_skills:

        if skill.lower() not in [
            s.lower() for s in skills
        ]:
            missing_skills.append(skill)

    # ==========================
    # ATS Score
    # ==========================

    ats_score = calculate_ats_score(
        skills
    )

    # ==========================
    # Suggestions
    # ==========================

    suggestions = get_suggestions(
        skills
    )

    # ==========================
    # Recommended Courses
    # ==========================

    courses = recommend_courses(
        missing_skills
    )

    # ==========================
    # Learning Roadmap
    # ==========================

    roadmap = generate_roadmap(
        role
    )

    # ==========================
    # Resume Match Percentage
    # ==========================

    if len(required_skills) > 0:

        match_percentage = int(
            (
                (len(required_skills) -
                 len(missing_skills))
                / len(required_skills)
            ) * 100
        )

    else:
        match_percentage = 0

    return render_template(
        "result.html",
        role=role,
        skills=skills,
        missing=missing_skills,
        courses=courses,
        ats_score=ats_score,
        suggestions=suggestions,
        roadmap=roadmap,
        match_percentage=match_percentage
    )

# ==========================
# Run Application
# ==========================

if __name__ == "__main__":
    app.run(debug=True)