"""
Manual Answer Generator for SemEval 2026 Task 12
Using document analysis and causal reasoning heuristics
"""

import json
from pathlib import Path
import zipfile
import re
from collections import defaultdict

# Paths
BASE_DIR = Path(r"d:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning")
TEST_DIR = BASE_DIR / "semeval2026-task12-dataset" / "test_data"
QUESTIONS_FILE = TEST_DIR / "questions.jsonl"
DOCS_FILE = TEST_DIR / "docs.json"
OUTPUT_FILE = BASE_DIR / "submission_manual.jsonl"
ZIP_FILE = BASE_DIR / "submission_manual.zip"


def load_documents():
    """Load and index documents by topic_id"""
    with open(DOCS_FILE, 'r', encoding='utf-8') as f:
        docs_data = json.load(f)
    
    # Index by topic_id
    docs_by_topic = {}
    for topic in docs_data:
        topic_id = topic['topic_id']
        topic_name = topic['topic']
        docs = topic.get('docs', [])
        
        # Combine all document content for this topic
        all_content = topic_name + " "
        for doc in docs:
            all_content += doc.get('title', '') + " " + doc.get('content', '') + " "
        
        docs_by_topic[topic_id] = {
            'topic': topic_name,
            'content': all_content.lower(),
            'docs': docs
        }
    
    return docs_by_topic


def load_questions():
    """Load questions from jsonl"""
    questions = []
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def clean_text(text):
    """Clean and normalize text for matching"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_key_phrases(text):
    """Extract key phrases from text"""
    words = clean_text(text).split()
    # Remove common words
    stopwords = {'the', 'a', 'an', 'is', 'was', 'were', 'are', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
                 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                 'between', 'under', 'again', 'further', 'then', 'once', 'and', 'or',
                 'but', 'if', 'because', 'until', 'while', 'although', 'that', 'which',
                 'who', 'whom', 'this', 'these', 'those', 'am', 'been', 'its', 'their',
                 'he', 'she', 'it', 'they', 'them', 'his', 'her', 'him', 'we', 'us',
                 'you', 'your', 'my', 'mine', 'our', 'ours', 'none', 'others', 'correct',
                 'causes', 'cause', 'caused', 'causing'}
    return [w for w in words if w not in stopwords and len(w) > 2]


def calculate_text_similarity(text1, text2):
    """Calculate simple word overlap similarity"""
    words1 = set(get_key_phrases(text1))
    words2 = set(get_key_phrases(text2))
    
    if not words1 or not words2:
        return 0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0


def is_temporal_cause(option_text, target_text, doc_content):
    """Check if option temporally precedes target in documents"""
    option_clean = clean_text(option_text)
    target_clean = clean_text(target_text)
    
    # Find positions in document
    option_pos = doc_content.find(option_clean[:50])  # First 50 chars
    target_pos = doc_content.find(target_clean[:50])
    
    if option_pos != -1 and target_pos != -1:
        return option_pos < target_pos
    return None


def find_best_answer(question, docs_by_topic):
    """Find the best answer using document analysis and causal reasoning"""
    topic_id = question['topic_id']
    target_event = question['target_event']
    
    options = {
        'A': question['option_A'],
        'B': question['option_B'],
        'C': question['option_C'],
        'D': question['option_D']
    }
    
    # Get relevant documents
    topic_data = docs_by_topic.get(topic_id, {'content': '', 'docs': []})
    doc_content = topic_data['content']
    
    # Check for "None of the others" option
    none_option = None
    for key, text in options.items():
        if 'none of the others' in text.lower():
            none_option = key
            break
    
    # Score each option
    scores = {}
    for key, option_text in options.items():
        if 'none of the others' in option_text.lower():
            scores[key] = -0.1  # Default low score for "none" option
            continue
        
        score = 0
        
        # 1. Text similarity with documents
        similarity = calculate_text_similarity(option_text, doc_content)
        score += similarity * 2
        
        # 2. Check if option appears in documents
        option_clean = clean_text(option_text)
        if option_clean[:30] in doc_content:
            score += 0.5
        
        # 3. Temporal reasoning - causes should appear before effects
        temporal = is_temporal_cause(option_text, target_event, doc_content)
        if temporal is True:
            score += 0.3
        elif temporal is False:
            score -= 0.2
        
        # 4. Keyword matching with target event
        target_similarity = calculate_text_similarity(option_text, target_event)
        # Some similarity is good (related), but too much might mean same event
        if 0.1 < target_similarity < 0.7:
            score += target_similarity * 0.5
        elif target_similarity > 0.8:
            score -= 0.3  # Too similar, might be same event not cause
        
        # 5. Causal keywords boost
        causal_words = ['launch', 'start', 'begin', 'announce', 'order', 'decision',
                        'cause', 'lead', 'result', 'trigger', 'due', 'after',
                        'following', 'struck', 'hit', 'attack', 'fire', 'kill',
                        'died', 'crash', 'earthquake', 'flood', 'storm']
        for word in causal_words:
            if word in option_clean:
                score += 0.05
        
        scores[key] = score
    
    # If all scores are very low, consider "none" option
    max_score = max(scores.values())
    if max_score < 0.05 and none_option:
        return none_option
    
    # Return option with highest score
    best = max(scores, key=scores.get)
    return best


def main():
    print("Loading documents...")
    docs_by_topic = load_documents()
    print(f"Loaded documents for {len(docs_by_topic)} topics")
    
    print("Loading questions...")
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")
    
    print("\nGenerating answers...")
    answers = []
    
    for i, question in enumerate(questions):
        answer = find_best_answer(question, docs_by_topic)
        answers.append({
            'id': question['id'],
            'answer': answer
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(questions)} questions...")
    
    # Save to jsonl
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for ans in answers:
            f.write(json.dumps(ans) + '\n')
    
    # Create zip
    print(f"Creating zip: {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUTPUT_FILE, 'submission.jsonl')
    
    # Statistics
    from collections import Counter
    answer_dist = Counter(a['answer'] for a in answers)
    print("\nAnswer distribution:")
    for key, count in sorted(answer_dist.items()):
        print(f"  {key}: {count} ({100*count/len(answers):.1f}%)")
    
    print(f"\n✅ Done! Submission files created:")
    print(f"   - {OUTPUT_FILE}")
    print(f"   - {ZIP_FILE}")


if __name__ == "__main__":
    main()
