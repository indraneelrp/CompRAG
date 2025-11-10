# save as: evaluate_simple.py
import ujson as json
import re
import string
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    ZERO_METRIC = (0, 0, 0)
    
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return ZERO_METRIC
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def evaluate_predictions(predictions_file):
    """Evaluate predictions in list format"""
    with open(predictions_file) as f:
        predictions = json.load(f)
    
    metrics = {
        'em': 0,
        'f1': 0,
        'precision': 0,
        'recall': 0,
        'total': 0,
        'errors': 0
    }
    
    for pred in predictions:
        if 'error' in pred or not pred.get('predicted_answer'):
            metrics['errors'] += 1
            continue
        
        predicted = pred['predicted_answer']
        gold = pred['gold_answer']
        
        em = exact_match_score(predicted, gold)
        f1, prec, rec = f1_score(predicted, gold)
        
        metrics['em'] += em
        metrics['f1'] += f1
        metrics['precision'] += prec
        metrics['recall'] += rec
        metrics['total'] += 1
    
    # Average
    n = metrics['total']
    if n > 0:
        metrics['em'] /= n
        metrics['f1'] /= n
        metrics['precision'] /= n
        metrics['recall'] /= n
    
    # Print
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total questions:  {len(predictions)}")
    print(f"Successful:       {metrics['total']}")
    print(f"Errors:           {metrics['errors']}")
    print("-"*60)
    print(f"Exact Match:      {metrics['em']:.4f} ({metrics['em']*100:.2f}%)")
    print(f"F1 Score:         {metrics['f1']:.4f}")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall:           {metrics['recall']:.4f}")
    print("="*60)
    
    return metrics

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_simple.py <predictions_file>")
        print("Example: python evaluate_simple.py predictions.json")
        sys.exit(1)
    
    evaluate_predictions(sys.argv[1])