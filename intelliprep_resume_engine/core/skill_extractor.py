_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def build_matcher(skill_list, nlp_inst):
    from spacy.matcher import PhraseMatcher
    matcher = PhraseMatcher(nlp_inst.vocab, attr="LOWER")
    patterns = [nlp_inst.make_doc(skill) for skill in skill_list]
    matcher.add("SKILLS", patterns)
    return matcher


def extract_skills(text, skill_list):
    nlp_inst = get_nlp()
    doc = nlp_inst(text)
    matcher = build_matcher(skill_list, nlp_inst)

    matches = matcher(doc)
    found_skills = set()

    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text.lower())

    return list(found_skills)
