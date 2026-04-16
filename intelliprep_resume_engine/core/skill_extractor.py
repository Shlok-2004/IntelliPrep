_nlp = None
_skill_extractor = None


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_skill_extractor():
    """Lazy-load the skillNer SkillExtractor once and reuse it."""
    global _skill_extractor
    if _skill_extractor is None:
        from spacy.matcher import PhraseMatcher
        from skillNer.general_params import SKILL_DB
        from skillNer.skill_extractor_class import SkillExtractor
        _skill_extractor = SkillExtractor(get_nlp(), SKILL_DB, PhraseMatcher)
    return _skill_extractor


def build_matcher(skill_list, nlp_inst):
    from spacy.matcher import PhraseMatcher
    matcher = PhraseMatcher(nlp_inst.vocab, attr="LOWER")
    patterns = [nlp_inst.make_doc(skill) for skill in skill_list]
    matcher.add("SKILLS", patterns)
    return matcher


def extract_skills(text, skill_list):
    """Match a fixed skill_list against text (used for resume vs JD skills)."""
    nlp_inst = get_nlp()
    doc = nlp_inst(text)
    matcher = build_matcher(skill_list, nlp_inst)

    matches = matcher(doc)
    found_skills = set()

    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text.lower())

    return list(found_skills)


def extract_skills_from_jd(jd_text):
    """
    Use skillNer to extract skills directly from the job description text.
    Returns a deduplicated list of skill name strings.
    Falls back to an empty list if extraction fails.
    """
    try:
        extractor = get_skill_extractor()
        annotations = extractor.annotate(jd_text)
        results = annotations.get("results", {})

        skills = set()
        for match in results.get("full_matches", []):
            skills.add(match["doc_node_value"].lower())
        for match in results.get("ngram_scored", []):
            skills.add(match["doc_node_value"].lower())

        return list(skills)
    except Exception:
        return []
