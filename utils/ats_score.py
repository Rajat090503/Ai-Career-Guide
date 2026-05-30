def calculate_ats_score(skills):

    score = 40

    score += len(skills) * 8

    if score > 100:
        score = 100

    return score