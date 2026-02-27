import json
from collections import defaultdict

with open('reference.jsonl', 'r') as f:
    refs = [json.loads(line) for line in f]
    
with open('exp32_32b_submission_extended_lora/submission.jsonl', 'r') as f:
    preds = [json.loads(line) for line in f]

ref_dict = {item['id']: item['answer'] for item in refs}

exact_matches = 0
partial_matches = 0
incorrect = 0
total = len(refs)
score = 0.0

fp_count = 0
fn_count = 0
total_predicted_labels = 0
total_gold_labels = 0

error_types = {
    'Missing Cause (False Negative)': 0,
    'Fabricated Cause (False Positive)': 0,
    'Failed to recognize "None"': 0,
    'Incorrectly predicted "None"': 0,
}

for pred in preds:
    q_id = pred['id']
    if q_id not in ref_dict:
        continue
    
    p_set = set(pred['answer'].split(',')) if pred['answer'] else set()
    g_set = set(ref_dict[q_id].split(',')) if ref_dict[q_id] else set()
    
    # "None of the others" is typically a specific option, e.g., 'D', but let's just analyze sets for now
    if p_set == g_set:
        exact_matches += 1
        score += 1.0
    elif p_set.issubset(g_set) and len(p_set) > 0:
        partial_matches += 1
        score += 0.5
        error_types['Missing Cause (False Negative)'] += 1
    else:
        incorrect += 1
        # Analyze why it's incorrect
        if len(p_set - g_set) > 0:
            error_types['Fabricated Cause (False Positive)'] += 1
        if len(g_set - p_set) > 0:
            error_types['Missing Cause (False Negative)'] += 1

print(f"Total: {total}")
print(f"Exact Matches: {exact_matches} ({(exact_matches/total)*100:.1f}%)")
print(f"Partial Matches: {partial_matches} ({(partial_matches/total)*100:.1f}%)")
print(f"Incorrect: {incorrect} ({(incorrect/total)*100:.1f}%)")
print(f"Average AER Score: {score/total:.3f}")
print("Error Breakdown:")
for k, v in error_types.items():
    print(f"  {k}: {v}")

# Write breakdown to a markdown file
with open('error_analysis_stats.md', 'w') as f:
    f.write(f"Exact Matches: {exact_matches}\n")
    f.write(f"Partial Matches: {partial_matches}\n")
    f.write(f"Incorrect: {incorrect}\n")
    f.write(f"Score: {score/total:.4f}\n")
    for k, v in error_types.items():
        f.write(f"  {k}: {v}\n")
