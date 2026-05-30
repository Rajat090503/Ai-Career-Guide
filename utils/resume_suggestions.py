def get_suggestions(skills):

    suggestions = []

    if len(skills) < 5:
        suggestions.append("Add more technical skills")

    suggestions.append("Add industry projects")
    suggestions.append("Add certifications")
    suggestions.append("Improve resume keywords")
    suggestions.append("Quantify achievements with numbers")
    suggestions.append("Add internship experience if available")

    return suggestions