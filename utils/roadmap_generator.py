import pandas as pd

roadmap_df = pd.read_csv("roadmaps.csv")

def generate_roadmap(role):

    role = role.strip()

    roadmap = roadmap_df[
        roadmap_df["role"] == role
    ]["step"].tolist()

    if len(roadmap) == 0:
        return [
            "Learn Core Skills",
            "Build Projects",
            "Prepare Resume",
            "Apply for Jobs"
        ]

    return roadmap