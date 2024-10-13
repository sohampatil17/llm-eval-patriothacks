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
        print(json.loads(response.content.decode()))
        return json.loads(response.content.decode())

# Using GPT-4 to evaluate the coherence of generated requirements
def evaluate_coherence(requirements):
    gpt4_url = "https://oss-arcfield-apim-prod.azure-api.net/patriothacks/hackathon/LLMCompletion"
    headers = {
        'x-auth': 'c2VjcmV0aGFja2F0aG9ucHdk',
        'Content-Type': 'application/json'
    }
    prompt = f"Rate the coherence of these requirements: {requirements}. Score between 1 (poor) and 5 (excellent)."
    data = {"prompt": prompt}
    response = requests.post(gpt4_url, headers=headers, data=json.dumps(data))
    return json.loads(response.content.decode())

# Example usage
requirements_data = extract_requirements('airforce.pdf')
requirements_text = " ".join([req['text'] for req in requirements_data['full_pydanctic_output']['requirements']])
coherence_score = evaluate_coherence(requirements_text)
print(f"Coherence Score: {coherence_score}")