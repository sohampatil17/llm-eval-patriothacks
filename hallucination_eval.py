from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import requests, json
from requests_toolbelt import MultipartEncoder

# Function to extract requirements from PDF using the LLM API
def extract_requirements(pdf_path):
    url = "https://oss-arcfield-apim-prod.azure-api.net/patriothacks/hackathon/ExtractRequirements"
    headers = {
        'x-auth': 'c2VjcmV0aGFja2F0aG9ucHdk',
    }
    with open(pdf_path, 'rb') as f:
        multi_part_data = MultipartEncoder(
            fields={"document": ("doc.pdf", f, "application/pdf")}
        )
        headers['Content-Type'] = multi_part_data.content_type
        response = requests.post(url, headers=headers, data=multi_part_data)
        return json.loads(response.content.decode())

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
    return {
        'differences': [],
        'similarity_score': similarity_score,
        'hallucination': False
    }

def detect_hallucination_with_explanation(pdf_path, iterations=3, similarity_threshold=0.85):
    print(f"\nRunning hallucination detection for {iterations} iterations...\n")
    
    results = []
    explanations = []
    
    # Run multiple iterations to extract requirements
    for iteration in range(1, iterations + 1):
        print(f"\nExtracting requirements (Iteration {iteration})...\n")
        result = extract_requirements(pdf_path)
        requirements_text = " ".join([req['text'] for req in result['full_pydanctic_output']['requirements']])
        results.append(requirements_text)
    
    # Compare each result with the first iteration
    base_result = results[0]
    hallucination_detected = False
    
    for i, result in enumerate(results[1:], 1):
        comparison_result = compare_requirements(base_result, result, similarity_threshold)
        explanations.append({
            'iteration': i+1,
            'hallucination': comparison_result['hallucination'],
            'similarity_score': comparison_result['similarity_score'],
            'differences': comparison_result['differences']
        })
        if comparison_result['hallucination']:
            hallucination_detected = True
    
    return {
        'hallucination_detected': hallucination_detected,
        'explanations': explanations
    }


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


detect_hallucination_with_explanation('airforce.pdf')