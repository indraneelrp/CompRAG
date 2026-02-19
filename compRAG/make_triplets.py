'''
Make triplets given chunks
Uses Babelscape/rebel-large (a seq2seq model) to extract relation triplets.
'''
import re
import torch
import spacy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from compRAG.text_samples import get_hardcoded_texts

# Load REBEL model and tokenizer onto GPU if available
_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large").to(_device)


def _parse_rebel_output(text):
    """Parse the REBEL model output into a list of (subject, relation, object) triplets.

    REBEL outputs triplets in the format:
        <triplet> subject <subj> tail <obj> relation
    where <subj> separates head from tail and <obj> separates tail from relation type.
    Multiple triplets sharing the same head are encoded as:
        <triplet> head <subj> tail1 <obj> rel1 <subj> tail2 <obj> rel2
    """
    triplets = []
    current = None
    subject = ''
    relation = ''
    object_ = ''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            # Save completed triplet before starting a new head
            if current == 'rel' and subject and relation and object_:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            current = 'subj'
            subject = ''
            relation = ''
            object_ = ''
        elif token == "<subj>":
            # Save completed triplet for shared-head case before reading next tail
            if current == 'rel' and subject and relation and object_:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            current = 'obj'   # tokens after <subj> are the TAIL entity
            object_ = ''
        elif token == "<obj>":
            current = 'rel'   # tokens after <obj> are the RELATION type
            relation = ''
        else:
            if current == 'subj':
                subject += ' ' + token
            elif current == 'obj':
                object_ += ' ' + token
            elif current == 'rel':
                relation += ' ' + token

    # Don't forget the last triplet
    if current == 'rel' and subject and relation and object_:
        triplets.append((subject.strip(), relation.strip(), object_.strip()))

    return triplets


def _extract_triplets_rebel(text):
    """Extract triplets from text using the REBEL model.

    Splits the text into sentences before passing to REBEL, since the model
    was trained on sentence-level inputs and performs poorly on full paragraphs.
    All sentences from a chunk are batched into a single generate() call so
    the GPU processes them in parallel rather than one at a time.
    """
    # Strip markdown formatting that confuses REBEL (bold, italic, links)
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)   # **bold** / *italic*
    text = re.sub(r'\[(.*?)\](?:\(.*?\))?', r'\1', text)  # [text](url) or [text]
    # Normalize whitespace — embedded newlines cause REBEL to produce garbled tokens
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return []

    # Tokenize all sentences at once and move to device
    inputs = _tokenizer(
        sentences,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(_device)

    # Single batched generate call: outputs shape = (len(sentences) * num_return_sequences, seq_len)
    # i.e. [seq0_sent0, seq1_sent0, seq2_sent0, seq0_sent1, seq1_sent1, seq2_sent1, ...]
    num_return_sequences = 3
    outputs = _model.generate(
        **inputs,
        max_length=512,
        num_beams=5,
        num_return_sequences=num_return_sequences,
    )

    decoded_all = _tokenizer.batch_decode(outputs, skip_special_tokens=False)

    all_triplets = []
    for sent_idx, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        # Slice the 3 sequences that belong to this sentence
        sent_decoded = decoded_all[sent_idx * num_return_sequences :
                                   (sent_idx + 1) * num_return_sequences]
        triplets = []
        for decoded in sent_decoded:
            triplets.extend(_parse_rebel_output(decoded))
        # Grounding check: drop hallucinations where neither entity appears in the source sentence
        triplets = [
            (s, r, o) for s, r, o in triplets
            if s.lower() in sentence_lower or o.lower() in sentence_lower
        ]
        all_triplets.extend(triplets)
    return all_triplets


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
        # Skip self-referential triplets (REBEL artifact where subject == object)
        if subj.lower() == obj.lower():
            continue
        # Skip clause subjects — REBEL sometimes extracts pronouns/clauses instead of entities
        # e.g. ('one had left the company', 'employer', 'Google')
        # Check first word: real entity names don't start with pronouns
        _pronouns = {'one', 'they', 'it', 'he', 'she', 'we', 'you', 'this', 'that', 'which', 'who'}
        if len(subj.split()) > 2 and subj.lower().split()[0] in _pronouns:
            continue

        # Normalize and deduplicate
        triplet = (subj.lower(), rel, obj.lower())
        if triplet not in seen:
            seen.add(triplet)
            cleaned.append((subj, rel, obj))

    return cleaned


def main_generate_triplets(nlp, text: str, is_query=False):
    """Extract triplets from text using REBEL, supplemented by spacy NER.

    Args:
        nlp: spacy model used for NER entity anchoring and query expansion.
        text: Input text to extract triplets from.
        is_query: If True, adds wildcard noun-chunk triplets for query matching.
    """
    triplets = _extract_triplets_rebel(text)

    if is_query and nlp is not None:
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
    nlp = spacy.load("en_core_web_sm")

    for i, t in enumerate(texts, 0):
        print(f"\n--- Text {i} ---")
        # triplets = main_generate_triplets(nlp, t)
        triplets = _extract_triplets_rebel(t)
        cleaned_triplets = clean_triplets(triplets)

        print(f"Raw triplets: {len(triplets)}")
        print(f"Cleaned triplets: {len(cleaned_triplets)}")

        # for triplet in triplets:
        #     print(f"  {triplet}")

        for triplet in cleaned_triplets:
            print(f"  {triplet}")
