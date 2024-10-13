import requests, json
from requests_toolbelt import MultipartEncoder
from rouge_score import rouge_scorer

# Extract requirements using the provided API
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

# Function to calculate ROUGE score
def calculate_rouge(source_text, generated_text):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(source_text, generated_text)
    
    # Extract relevant scores
    rouge1_recall = scores['rouge1'].recall
    rouge2_recall = scores['rouge2'].recall
    rougeL_recall = scores['rougeL'].recall
    
    return {
        'rouge1_recall': rouge1_recall,
        'rouge2_recall': rouge2_recall,
        'rougeL_recall': rougeL_recall
    }

# Function to evaluate content coverage
def evaluate_content_coverage(pdf_path):
    # Extract requirements
    try:
        requirements_data = extract_requirements(pdf_path)
    except Exception as e:
        print(f"Error extracting requirements: {e}")
        return []
    
    if not requirements_data or 'full_pydanctic_output' not in requirements_data or 'requirements' not in requirements_data['full_pydanctic_output']:
        print("No valid requirements found in the API response.")
        return []

    requirements = requirements_data['full_pydanctic_output']['requirements']
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    coverage_results = []
    
    for req in requirements:
        source_text = req.get('source', {}).get('source text', '')
        generated_text = req.get('text', '')
        
        if source_text and generated_text:
            scores = scorer.score(source_text, generated_text)
            
            coverage_results.append({
                'requirement': generated_text[:100] + "...",  # Truncate for display
                'source': source_text[:100] + "...",  # Truncate for display
                'rouge1_recall': scores['rouge1'].recall,
                'rouge2_recall': scores['rouge2'].recall,
                'rougeL_recall': scores['rougeL'].recall
            })
    
    return coverage_results
# Example usage
#evaluate_content_coverage('airforce.pdf')
