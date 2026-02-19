from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def skill_match_score(found_skills, role_skills):
    found = set([s.lower() for s in found_skills])
    required = set([s.lower() for s in role_skills])

    matched = list(found.intersection(required))
    missing = list(required.difference(found))

    score = (len(matched) / len(required)) * 100 if required else 0

    return round(score, 2), matched, missing


def semantic_similarity(resume_text, jd_text):
    documents = [resume_text, jd_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return round(float(similarity[0][0]) * 100, 2)


def final_ats_score(skill_score, semantic_score, w1=0.6, w2=0.4):
    return round((w1 * skill_score) + (w2 * semantic_score), 2)


def critical_skill_gaps(found_skills, critical_skills):
    found = set([s.lower() for s in found_skills])
    critical = set([s.lower() for s in critical_skills])

    missing_critical = list(critical.difference(found))
    present_critical = list(critical.intersection(found))

    return present_critical, missing_critical
