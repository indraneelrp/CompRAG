'''
Make triplets given chunks
2 methods: 
    a. SpaCy dependency parse (rule-based) + spacy model
    b. just a small LM (bert)
'''
import spacy


def make_triplets(chunk_id: int) -> list[str]:
    return []

def save_triplets(triple_list: list, chunk_id: int) -> None:
    # save to SQLite or something
    pass


# =========================================================================================
# ****  Some harcoded functions as initial work, to be refactored into proper pipeline  *****
# =========================================================================================

# Load  model (make sure it's installed: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# Example text from hotpot qa
text = """Mother Love Bone was an American rock band that
formed in Seattle, Washington in 1987. The band was active from 1987 to 1990. Frontman Andrew
Wood’s personality and compositions helped to catapult the group to the top of the burgeoning late 1980s/early
1990s Seattle music scene. Wood died only days before the scheduled release of the band’s debut album,
'Apple', thus ending the group’s hopes of success."""

text1 = '''**Eight names are** listed as authors on “Attention Is All You Need,” a scientific paper written in the spring of 2017. They were all [Google] researchers, though by then one had left the company. When the most tenured contributor, Noam Shazeer, saw an early draft, he was surprised that his name appeared first, suggesting his contribution was paramount. “I wasn’t thinking about it,” he says.
'''

text2 = '''
It’s always a delicate balancing act to figure out how to list names—who gets the coveted lead position, who’s shunted to the rear. Especially in a case like this one, where each participant left a distinct mark in a true group effort. As the researchers hurried to finish their paper, they ultimately decided to “sabotage” the convention of ranking contributors. They added an asterisk to each name and a footnote: “Equal contributor,” it read. “Listing order is random.” The writers sent the paper off to a prestigious artificial intelligence conference just before the deadline—and kicked off a revolution.
'''

text3= '''
But the field was running into limitations. Recurrent neural networks struggled to parse longer chunks of text. Take a passage like *Joe is a baseball player, and after a good breakfast he went to the park and got two hits.* To make sense of “two hits,” a language model has to remember the part about baseball. In human terms, it has to be paying attention. The accepted fix was something called “long short-term memory” (LSTM), an innovation that allowed language models to process bigger and more complex sequences of text. But the computer still handled those sequences strictly sequentially—word by tedious word—and missed out on context clues that might appear later in a passage. “The methods we were applying were basically Band-Aids,” Uszkoreit says. “We could not get the right stuff to really work at scale.”
'''

text4= '''
Around 2014, he began to concoct a different approach that he referred to as self-attention. This kind of network can translate a word by referencing *any* other part of a passage. Those other parts can clarify a word’s intent and help the system produce a good translation. “It actually considers everything and gives you an efficient way of looking at many inputs at the same time and then taking something out in a pretty selective way,” he says. Though AI scientists are careful not to confuse the metaphor of neural networks with the way the biological brain actually works, Uszkoreit does seem to believe that self-attention is somewhat similar to the way humans process language.
'''

text5 = '''
Uszkoreit thought a self-attention model could potentially be faster and more effective than recurrent neural nets. The way it handles information was also perfectly suited to the powerful parallel processing chips that were being produced en masse to support the machine learning boom. Instead of using a linear approach (look at every word in sequence), it takes a more parallel one (look at a bunch of them together). If done properly, Uszkoreit suspected, you could use self-attention *exclusively* to get better results.

Not everyone thought this idea was going to rock the world, including Uszkoreit’s father, who had scooped up two Google Faculty research awards while his son was working for the company. “People raised their eyebrows, because it dumped out all the existing neural architectures,” Jakob Uszkoreit says. Say goodbye to recurrent neural nets? Heresy! “From dinner-table conversations I had with my dad, we weren’t necessarily seeing eye to eye.”
'''

text6 ='''
Anarchism is a political philosophy that advocates self-governed societies based on voluntary institutions. These are often described as stateless societies, although several authors have defined them more specifically as institutions based on non-hierarchical free associations. Anarchism holds the state to be undesirable, unnecessary and harmful
'''

def get_full_phrase(token, direction="both"):
    """Extract full noun phrases by following compound and modifier relationships."""
    tokens = [token]
    
    if direction in ["left", "both"]:
        # Add left modifiers (adjectives, compounds, etc.)
        for left_token in token.lefts:
            if left_token.dep_ in ("compound", "amod", "det", "poss"):
                tokens.insert(0, left_token)
    
    if direction in ["right", "both"]:
        # Add right modifiers
        for right_token in token.rights:
            if right_token.dep_ in ("compound", "amod"):
                tokens.append(right_token)
    
    return " ".join([t.text for t in tokens])

def find_subjects(verb):
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
    
    return objects

def extract_triplets(doc):
    """Extract subject-relation-object triplets from spaCy doc."""
    triplets = []
    
    for sent in doc.sents:
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
        if len(subj) < 2 or len(obj) < 2:
            continue
        if any(char in subj + obj for char in ['*', '[', ']']):
            continue
            
        # Normalize and deduplicate
        triplet = (subj.lower(), rel, obj.lower())
        if triplet not in seen:
            seen.add(triplet)
            cleaned.append((subj, rel, obj))
    
    return cleaned


def main_generate_triplets_hardcoded():
    texts = [text, text1, text2, text3, text4]
    out = []
    
    for i, t in enumerate(texts, 1):
        doc = nlp(t)
        triplets = extract_triplets(doc)
        cleaned_triplets = clean_triplets(triplets)
        out.append(cleaned_triplets)
    
    return out


def main_generate_triplets(nlp, text:str)-> list[tuple[str, str, str]]:
    doc = nlp(text)
    triplets = extract_triplets(doc)
    return clean_triplets(triplets)

if __name__ == "__main__":
    texts = [text, text1, text2, text4, text5, text6]
    
    for i, t in enumerate(texts, 1):
        print(f"\n--- Text {i} ---")
        doc = nlp(t)
        triplets = extract_triplets(doc)
        cleaned_triplets = clean_triplets(triplets)
        
        print(f"Raw triplets: {len(triplets)}")
        print(f"Cleaned triplets: {len(cleaned_triplets)}")
        
        for triplet in cleaned_triplets:
            print(f"  {triplet}")

