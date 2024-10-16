from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import json

# Compare two requirement texts and provide an explanation with a similarity threshold
def compare_requirements(req1, req2, similarity_threshold=0.85):
    print("\nComparing requirements:")
    print(f"Requirement 1: {req1[:100]}...")  # Print first 100 chars for clarity
    print(f"Requirement 2: {req2[:100]}...\n")
    
    # Compute sequence differences using difflib's SequenceMatcher
    matcher = SequenceMatcher(None, req1, req2)
    differences = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':  # If the segments are not equal
            differences.append({
                'change': tag, 
                'from': req1[i1:i2], 
                'to': req2[j1:j2]
            })

    print("Computed text differences...")
    
    # Compute semantic similarity between the two requirements
    similarity_score = compute_semantic_similarity(req1, req2)
    print(f"Semantic Similarity Score: {similarity_score:.2f}\n")
    
    # If similarity is below the threshold, highlight the hallucination
    if similarity_score < similarity_threshold:
        print(f"Similarity below threshold ({similarity_threshold}), possible hallucination detected.\n")
        return {
            'differences': differences,
            'similarity_score': similarity_score,
            'hallucination': True
        }
    return {'hallucination': False, 'similarity_score': similarity_score, 'differences': []}

# Function to compute semantic similarity using BERT embeddings
def compute_semantic_similarity(text1, text2):
    print("Computing semantic similarity using BERT...")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs1 = tokenizer(text1, return_tensors='pt')
    inputs2 = tokenizer(text2, return_tensors='pt')

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # Take the mean of the embeddings
    embedding1 = torch.mean(outputs1.last_hidden_state, dim=1)
    embedding2 = torch.mean(outputs2.last_hidden_state, dim=1)

    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    
    print(f"Cosine similarity computed: {similarity[0][0]:.2f}")
    return similarity[0][0]

# Detect hallucinations and explain differences
def detect_hallucination(json_data, similarity_threshold=0.5):
    requirements = json_data['full_pydanctic_output']['requirements']
    source_texts = [req["source"]["source text"] for req in requirements]
    generated_texts = [req["text"] for req in requirements]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(source_texts + generated_texts)
    
    similarity_scores = cosine_similarity(tfidf_matrix[:len(source_texts)], tfidf_matrix[len(source_texts):])
    
    hallucination_results = []
    hallucination_detected = False
    
    for i, score in enumerate(similarity_scores.diagonal()):
        is_hallucination = score < similarity_threshold
        hallucination_results.append({
            "iteration": i + 1,
            "hallucination": is_hallucination,
            "similarity_score": float(score)  # Convert numpy float to Python float
        })
        if is_hallucination:
            hallucination_detected = True
    
    return {
        "hallucination_detected": hallucination_detected,
        "explanations": hallucination_results
    }
    
# Example usage
if __name__ == "__main__":
    with open('path/to/your/json/file.json', 'r') as f:
        json_data = json.load(f)
    hallucination_results = detect_hallucination(json_data)
    print(json.dumps(hallucination_results, indent=2))