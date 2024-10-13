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
            fields={"document": ("airforce.pdf", f, "application/pdf")}
        )
        headers['Content-Type'] = multi_part_data.content_type
        response = requests.post(url, headers=headers, data=multi_part_data)
        
    # Print the full response to inspect the structure
    response_content = json.loads(response.content.decode())
    print(json.dumps(response_content, indent=2))
    
    # Handle typo in the key
    if 'full_pydanctic_output' in response_content:
        return response_content['full_pydanctic_output']['requirements']
    elif 'requirements' in response_content:
        return response_content['requirements']
    else:
        raise KeyError("Error: 'full_pydantic_output' or 'requirements' key not found in the API response.")

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
    except KeyError as e:
        print(f"Error extracting requirements: {e}")
        return None
    
    coverage_results = []
    
    # Check if requirements_data is a dictionary and has the expected structure
    if isinstance(requirements_data, dict) and 'full_pydanctic_output' in requirements_data:
        requirements = requirements_data['full_pydanctic_output'].get('requirements', [])
    elif isinstance(requirements_data, list):
        requirements = requirements_data
    else:
        print("Unexpected format of requirements data")
        return None

    # Iterate through extracted requirements
    for req in requirements:
        source_text = req.get('source', {}).get('source text', '')
        generated_text = req.get('text', '')
        
        if source_text and generated_text:
            # Calculate ROUGE scores
            rouge_scores = calculate_rouge(source_text, generated_text)
            
            coverage_results.append({
                'requirement': generated_text[:100] + "...",  # Truncate for display
                'source': source_text[:100] + "...",  # Truncate for display
                'rouge1_recall': rouge_scores['rouge1_recall'],
                'rouge2_recall': rouge_scores['rouge2_recall'],
                'rougeL_recall': rouge_scores['rougeL_recall']
            })
    
    return coverage_results

# Example usage
evaluate_content_coverage('airforce.pdf')
