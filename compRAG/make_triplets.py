'''
Make triplets given chunks
Uses Babelscape/rebel-large (a seq2seq model) to extract relation triplets.
'''
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from compRAG.text_samples import get_hardcoded_texts
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load REBEL model and tokenizer
_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")


def _parse_rebel_output(text):
    """Parse the REBEL model output into a list of (subject, relation, object) triplets.

    REBEL outputs triplets in the format:
        <triplet> subject <subj> relation <obj> object
    """
    triplets = []
    current = None
    subject = ''
    relation = ''
    object_ = ''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            if current == 'obj' and subject and relation and object_:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            current = 'subj'
            subject = ''
            relation = ''
            object_ = ''
        elif token == "<subj>":
            current = 'rel'
            relation = ''
        elif token == "<obj>":
            current = 'obj'
            object_ = ''
        else:
            if current == 'subj':
                subject += ' ' + token
            elif current == 'obj':
                object_ += ' ' + token
            elif current == 'rel':
                relation += ' ' + token

    # Don't forget the last triplet
    if current == 'obj' and subject and relation and object_:
        triplets.append((subject.strip(), relation.strip(), object_.strip()))

    return triplets


def _extract_triplets_rebel(text):
    """Extract triplets from text using the REBEL model."""
    inputs = _tokenizer(
        text,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    outputs = _model.generate(
        **inputs,
        max_length=256,
        num_beams=3,
        num_return_sequences=1,
    )
    decoded = _tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    return _parse_rebel_output(decoded)


def clean_triplets(triplets):
    """Remove duplicate and low-quality triplets."""
    cleaned = []
    seen = set()

    for subj, rel, obj in triplets:
        # Skip if too short or contains unwanted characters
        if len(subj) < 2 or len(rel) < 2:
            continue
        if len(obj) < 2 and obj != "?":
            continue
        if any(char in subj + obj for char in ['*', '[', ']']):
            continue

        # Normalize and deduplicate
        triplet = (subj.lower(), rel, obj.lower())
        if triplet not in seen:
            seen.add(triplet)
            cleaned.append((subj, rel, obj))

    if len(cleaned) < 3:
        return cleaned

    # tf-idf based cleaning (clean based on frequent RELATIONS ie the r in s,r,o)
    relations = [c[1].lower() for c in cleaned]
    vectorizer = TfidfVectorizer(analyzer='word', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(relations)

    avg_scores = tfidf_matrix.mean(axis=1).A1
    threshold = np.percentile(avg_scores, 60)
    key_triplets = [cleaned[i] for i, score in enumerate(avg_scores) if score >= threshold]

    return key_triplets


def main_generate_triplets(nlp, text: str, is_query=False):
    """Extract triplets from text using REBEL.

    Args:
        nlp: Kept for backward compatibility (unused).
        text: Input text to extract triplets from.
        is_query: If True, adds wildcard triplets for entities/noun chunks
                  to improve query matching.
    """
    triplets = _extract_triplets_rebel(text)

    if is_query:
        # Use spacy (if provided) to add wildcard entity triplets for query matching
        if nlp is not None:
            doc = nlp(text)
            for ent in doc.ents:
                if len(ent.text) > 2:
                    triplets.append((ent.text, "relates_to", "?"))
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 2 and chunk.text.lower() not in [
                    "who", "what", "which", "the tutor", "the person"
                ]:
                    triplets.append((chunk.text, "about", "?"))

    return clean_triplets(triplets)


def main_generate_triplets_from_list(nlp, textlist):
    """Extract triplets from a list of texts using REBEL.

    Args:
        nlp: Kept for backward compatibility (unused).
        textlist: List of input texts.
    """
    out = []
    for t in textlist:
        triplets = _extract_triplets_rebel(t)
        cleaned_triplets = clean_triplets(triplets)
        out.append(cleaned_triplets)
    return out


if __name__ == "__main__":
    texts = get_hardcoded_texts()

    for i, t in enumerate(texts, 1):
        print(f"\n--- Text {i} ---")
        triplets = _extract_triplets_rebel(t)
        cleaned_triplets = clean_triplets(triplets)

        print(f"Raw triplets: {len(triplets)}")
        print(f"Cleaned triplets: {len(cleaned_triplets)}")

        for triplet in cleaned_triplets:
            print(f"  {triplet}")
