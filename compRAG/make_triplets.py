'''
Make triplets given chunks
2 methods: 
    a. SpaCy dependency parse (rule-based) + spacy model
    b. just a small LM (bert) (not done yet)
'''
import spacy
from compRAG.text_samples import get_hardcoded_texts

# Load  model (make sure it's installed: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")


def get_full_phrase(token, direction="both", max_depth=3):
    """Extract full noun phrases by following compound and modifier relationships."""
    if max_depth <= 0:
        return token.text
        
    tokens = []
    
    if direction in ["left", "both"]:
        # Collect left modifiers in order
        left_modifiers = []
        for left_token in token.lefts:
            if left_token.dep_ in ("compound", "amod", "det", "poss", "nummod", "advmod"):
                left_modifiers.append(left_token)
        # Sort by position in sentence to maintain word order
        left_modifiers.sort(key=lambda x: x.i)
        tokens.extend(left_modifiers)
    
    # Add the head token
    tokens.append(token)
    
    if direction in ["right", "both"]:
        # Collect right modifiers in order
        right_modifiers = []
        for right_token in token.rights:
            if right_token.dep_ in ("compound", "amod", "nummod", "advmod"):
                right_modifiers.append(right_token)
        # Sort by position in sentence to maintain word order
        right_modifiers.sort(key=lambda x: x.i)
        tokens.extend(right_modifiers)

    return " ".join([t.text for t in tokens])


def find_subjects(verb, sent_root=None):
    """Find all subjects for a given verb."""
    subjects = []
    
    for token in verb.lefts:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subjects.append(get_full_phrase(token))
        elif token.dep_ == "csubj":  # Clausal subject
            subjects.append(token.text)
    
    # Handle cases where subject is in a different part of the sentence
    if not subjects and verb.head != verb:
        for token in verb.head.lefts:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subjects.append(get_full_phrase(token))

    # For questions, look for subjects in the entire sentence
    if not subjects and sent_root:
        for token in sent_root.subtree:
            if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                subjects.append(get_full_phrase(token))
    
    return subjects


def find_objects(verb):
    """Find all objects for a given verb."""
    objects = []
    
    for token in verb.rights:
        if token.dep_ in ("dobj", "attr", "pcomp", "acomp"):
            objects.append(get_full_phrase(token))
        elif token.dep_ == "xcomp":  # open clausal complement (like "to be undesirable")
            # look for the actual predicate in the xcomp
            for xcomp_child in token.subtree:
                if xcomp_child.dep_ in ("attr", "acomp") or (xcomp_child.pos_ == "ADJ"):
                    objects.append(get_full_phrase(xcomp_child))
        elif token.dep_ == "ccomp":  # clausal complement
            for ccomp_child in token.subtree:
                if ccomp_child.dep_ in ("attr", "acomp", "dobj") or (ccomp_child.pos_ == "ADJ"):
                    objects.append(get_full_phrase(ccomp_child))
        elif token.dep_ == "prep":
            for prep_child in token.children:
                if prep_child.dep_ == "pobj":
                    objects.append(get_full_phrase(prep_child))
        elif token.dep_ == "advmod" and token.pos_ == "ADV":
            objects.append(token.text)
    
    return objects


def extract_dependency_triplets(sent):
    """Extract triplets based on dependency relationships regardless of POS."""
    triplets = []
    
    for token in sent:
        # Skip punctuation and some function words
        if token.is_punct or token.is_space:
            continue
            
        # Look for meaningful dependency relationships
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                # Subject relationship
                subject = get_full_phrase(child)
                predicate = get_full_phrase(token)
                
                # Find objects of the predicate
                for obj_child in token.children:
                    if obj_child.dep_ in ("dobj", "attr", "pcomp", "advmod"):
                        obj = get_full_phrase(obj_child)
                        if subject.strip() and obj.strip():
                            triplets.append((subject.strip(), token.lemma_, obj.strip()))
    
    return triplets


def extract_question_triplets(sent):
    """Extract triplets specifically from question sentences."""
    triplets = []
    
    # Find the main verb in the question
    main_verbs = [token for token in sent if token.pos_ == "VERB" and not token.is_stop]
    
    for verb in main_verbs:
        subjects = find_subjects(verb, sent.root)
        
        # For questions, also look for objects in the broader sentence context
        objects = find_objects(verb)
        
        # Look for prepositional phrases that might contain important information
        for token in sent:
            if token.dep_ == "prep" and token.head == verb:
                for child in token.children:
                    if child.dep_ == "pobj":
                        prep_phrase = f"{token.text} {get_full_phrase(child)}"
                        objects.append(prep_phrase)
        
        # Create triplets
        for subj in subjects:
            for obj in objects:
                if subj.strip() and obj.strip():
                    relation = verb.lemma_
                    triplets.append((subj.strip(), relation, obj.strip()))
    
    return triplets


def extract_triplets(doc):
    """Extract subject-relation-object triplets from spaCy doc."""
    triplets = []
    
    for sent in doc.sents:
        is_question = any(token.text in ["what", "who", "where", "when", "why", "how", "which"] 
                         for token in sent[:3]) or sent.text.strip().endswith("?")
        
        if is_question:
            # Use specialized question handling
            question_triplets = extract_question_triplets(sent)
            triplets.extend(question_triplets)

        dep_triplets = extract_dependency_triplets(sent)
        triplets.extend(dep_triplets)

        for token in sent:
            # Look for verbs as potential relations
            if token.pos_ == "VERB" and not token.is_stop:
                subjects = find_subjects(token)
                objects = find_objects(token)
                
                # Create triplets for each subject-object combination
                for subj in subjects:
                    for obj in objects:
                        if subj.strip() and obj.strip():
                            relation = token.lemma_
                            triplets.append((subj.strip(), relation, obj.strip()))
            
            # Also look for copular constructions (is, was, etc.)
            elif token.lemma_ in ("be", "have") and token.pos_ == "AUX":
                subjects = find_subjects(token)
                objects = find_objects(token)
                
                for subj in subjects:
                    for obj in objects:
                        if subj.strip() and obj.strip():
                            triplets.append((subj.strip(), token.lemma_, obj.strip()))
    
    return triplets


def clean_triplets(triplets):
    """Remove duplicate and low-quality triplets."""
    cleaned = []
    seen = set()
    
    for subj, rel, obj in triplets:
        # Skip if too short or contains unwanted characters
        if len(subj) < 2 or len(obj) < 2 or len(rel)<2:
            continue
        if any(char in subj + obj for char in ['*', '[', ']']):
            continue
            
        # Normalize and deduplicate
        triplet = (subj.lower(), rel, obj.lower())
        if triplet not in seen:
            seen.add(triplet)
            cleaned.append((subj, rel, obj))
    
    return cleaned



def main_generate_triplets(nlp, text:str)-> list[tuple[str, str, str]]:
    doc = nlp(text)
    triplets = extract_triplets(doc)
    return clean_triplets(triplets)


def main_generate_triplets_from_list(nlp, textlist):
    out = []
    
    for i, t in enumerate(textlist, 1):
        doc = nlp(t)
        triplets = extract_triplets(doc)
        cleaned_triplets = clean_triplets(triplets)
        out.append(cleaned_triplets)
    
    return out


if __name__ == "__main__":
    texts = get_hardcoded_texts()
    
    for i, t in enumerate(texts, 1):
        print(f"\n--- Text {i} ---")
        doc = nlp(t)
        triplets = extract_triplets(doc)
        cleaned_triplets = clean_triplets(triplets)
        
        print(f"Raw triplets: {len(triplets)}")
        print(f"Cleaned triplets: {len(cleaned_triplets)}")
        
        for triplet in cleaned_triplets:
            print(f"  {triplet}")