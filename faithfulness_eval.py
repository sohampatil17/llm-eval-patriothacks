import requests, json
from requests_toolbelt import MultipartEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Extract requirements using the provided API
def extract_requirements(pdf_path):
    url = "https://oss-arcfield-apim-prod.azure-api.net/patriothacks/hackathon/ExtractRequirements"
    headers = {
        'x-auth': 'c2VjcmV0aGFja2F0aG9ucHdk',
    }
    with open(pdf_path, 'rb') as f:
        multi_part_data = MultipartEncoder(
            fields={"document": ("airforce.pdf", f, "application/pdf")}
        )
        headers['Content-Type'] = multi_part_data.content_type
        response = requests.post(url, headers=headers, data=multi_part_data)
    return json.loads(response.content.decode())

# Compute semantic similarity using BERT embeddings
def compute_semantic_similarity(text1, text2):
    print("Computing semantic similarity using BERT...")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs1 = tokenizer(text1, return_tensors='pt', truncation=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    embedding1 = torch.mean(outputs1.last_hidden_state, dim=1)
    embedding2 = torch.mean(outputs2.last_hidden_state, dim=1)

    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    
    print(f"Cosine similarity: {similarity[0][0]:.2f}")
    return similarity[0][0]

# Evaluate faithfulness for extraction
def evaluate_faithfulness_extraction(requirement, source_text, threshold=0.8):
    print("\nEvaluating faithfulness for extracted requirement...")
    similarity_score = compute_semantic_similarity(requirement, source_text)
    if similarity_score >= threshold:
        return {"faithful": True, "similarity_score": similarity_score}
    return {"faithful": False, "similarity_score": similarity_score}

# Detect faithfulness for extracted requirements
def detect_faithfulness_extraction(pdf_path, threshold=0.8):
    print("\nRunning faithfulness evaluation for extracted requirements...\n")
    result = extract_requirements(pdf_path)
    
    faithfulness_results = []
    for req in result['full_pydanctic_output']['requirements']:
        requirement_text = req['text']
        source_text = req['source']['source text']
        faithfulness = evaluate_faithfulness_extraction(requirement_text, source_text, threshold)
        faithfulness_results.append({
            'requirement': requirement_text[:100] + "...",  # Truncate for display
            'source': source_text[:100] + "...",  # Truncate for display
            'faithful': faithfulness['faithful'],
            'similarity_score': faithfulness['similarity_score']
        })
    
    return faithfulness_results
# Example usage for extraction faithfulness
detect_faithfulness_extraction('/Users/I212229/Desktop/Projects/llm-eval-patriothacks/extraction/airforce.pdf')


