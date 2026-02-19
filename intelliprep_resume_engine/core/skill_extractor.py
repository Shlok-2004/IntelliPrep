import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")


def build_matcher(skill_list):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SKILLS", patterns)
    return matcher


def extract_skills(text, skill_list):
    doc = nlp(text)
    matcher = build_matcher(skill_list)

    matches = matcher(doc)
    found_skills = set()

    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text.lower())

    return list(found_skills)
