import ujson as json
from datasets import load_dataset

def prepare_eval_dataset(output_file="hotpot_eval.json", split="validation", limit=None):
    """
    Extract questions and answers from HotpotQA validation set.
    """
    print(f"Loading HotpotQA {split} set...")
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")[split]
    
    eval_data = []
    
    for item in ds:
        eval_data.append({
            'id': item['id'],
            'question': item['question'],
            'answer': item['answer'],
            'type': item['type'],
            'level': item['level']
        })
        
        if limit and len(eval_data) >= limit:
            break
    
    with open(output_file, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"Saved {len(eval_data)} questions to {output_file}")
    return eval_data

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Small test set
        prepare_eval_dataset("hotpot_eval_test.json", limit=10)
    else:
        # Full validation set
        prepare_eval_dataset("hotpot_eval_full.json")