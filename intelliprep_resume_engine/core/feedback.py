def generate_feedback(role, skill_score, semantic_score, matched, missing, missing_critical):
    feedback = []

    if missing_critical:
        crit = ", ".join(missing_critical)
        feedback.append(
            f"Your resume is missing critical skills for this role, such as {crit}. These are essential for shortlisting."
        )

    if skill_score > 70:
        feedback.append("Your resume shows strong alignment with the selected role.")
    elif skill_score > 40:
        feedback.append("Your resume shows moderate alignment with the selected role.")
    else:
        feedback.append("Your resume currently shows low alignment with the selected role.")

    if missing:
        top_missing = ", ".join(missing[:5])
        feedback.append(
            f"Your resume is missing several core skills required for this role, such as {top_missing}."
        )

    feedback.append(
        "Adding more relevant technical keywords and detailed project-specific skills can significantly improve your ATS score."
    )

    feedback.append(
        "Consider adding numerical metrics in your projects (for example: accuracy improved by 12%, reduced processing time by 30%) to clearly demonstrate impact."
    )

    if semantic_score < 30:
        feedback.append(
            "Your resume content has low similarity with the job description. Try aligning your project descriptions with the role requirements."
        )

    return feedback
