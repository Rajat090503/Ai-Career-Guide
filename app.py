from flask import Flask, render_template, request
import os
import pickle
import pandas as pd

from utils.resume_parser import extract_text_from_pdf
from utils.skill_extractor import extract_skills
from utils.course_recommender import recommend_courses
from utils.ats_score import calculate_ats_score
from utils.resume_suggestions import get_suggestions
from utils.roadmap_generator import generate_roadmap

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load Skills Dataset
skills_df = pd.read_csv("roles_skills.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():

    try:
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

        # Predict Role
        X = vectorizer.transform([text])

        prediction = model.predict(X)

        print("Prediction:", prediction)

        role = str(prediction[0])

        print("Predicted Role:", role)

        # Required Skills
        required_skills = skills_df[
            skills_df["role"] == role
        ]["skill"].tolist()

        # Missing Skills
        missing_skills = []

        for skill in required_skills:
            if skill.lower() not in [s.lower() for s in skills]:
                missing_skills.append(skill)

        ats_score = calculate_ats_score(skills)

        suggestions = get_suggestions(skills)

        courses = recommend_courses(missing_skills)

        roadmap = generate_roadmap(role)

        if len(required_skills) > 0:
            match_percentage = int(
                (
                    (len(required_skills) - len(missing_skills))
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

    except Exception as e:
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
