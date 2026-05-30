skills_db = [
    "python",
    "sql",
    "excel",
    "power bi",
    "tableau",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pandas",
    "numpy",
    "statistics",
    "html",
    "css",
    "javascript",
    "react",
    "nodejs",
    "mongodb",
    "java",
    "spring boot",
    "mysql"
]

def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in skills_db:
        if skill in text:
            found_skills.append(skill)

    return list(set(found_skills))