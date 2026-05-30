import pandas as pd

courses_df = pd.read_csv("courses.csv")

def recommend_courses(missing_skills):

    recommendations = {}

    for skill in missing_skills:

        skill = skill.lower().strip()

        row = courses_df[
            courses_df["skill"].str.lower() == skill
        ]

        if not row.empty:
            recommendations[skill] = row.iloc[0]["course"]

    return recommendations