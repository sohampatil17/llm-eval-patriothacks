import requests
import json
import re

# LLM API URL and headers
url = 'https://oss-arcfield-apim-prod.azureapi.net/patriothacks/hackathon/LLMCompletion'
hdr = {
    'x-auth': 'c2VjcmV0aGFja2F0aG9ucHdk',
    'Content-Type': 'application/json',
    'Cache-Control': 'no-cache',
}

# EARS regex patterns
ears_patterns = {
    "Ubiquitous": re.compile(r".*shall.*"),
    "State_Driven": re.compile(r"While .*shall.*"),
    "Event_Driven": re.compile(r"When .*shall.*"),
    "Optional_Feature": re.compile(r"Where .*shall.*"),
    "Unwanted_Behavior": re.compile(r"If .*then .*shall.*"),
    "Complex": re.compile(r"While .*When .*shall.*")
}

# LLM function to suggest EARS pattern for non-EARS classification
def suggest_ears_pattern(requirement_text):
    try:
        prompt = f"Suggest a suitable EARS pattern (Ubiquitous, State-Driven, Event-Driven, etc.) for the following requirement:\n\n'{requirement_text}'"
        data = {"prompt": prompt}
        response = requests.post(url, headers=hdr, data=json.dumps(data))
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get('choices', [{}])[0].get('message', {}).get('content', 'No suggestion available')
        else:
            return f"Error in LLM request: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to assess compatibility with EARS
def evaluate_ears_compatibility(requirement_text):
    # Check for pattern match
    for pattern_name, pattern in ears_patterns.items():
        if pattern.match(requirement_text):
            return pattern_name
    return "Non-EARS"

# Function to provide overall LLM analysis
def overall_llm_explanation(requirements_summary):
    try:
        prompt = f"Based on the following summary of requirements and their classification, explain any areas of improvement and missing patterns:\n\n{requirements_summary}"
        data = {"prompt": prompt}
        response = requests.post(url, headers=hdr, data=json.dumps(data))
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get('choices', [{}])[0].get('message', {}).get('content', 'No explanation available')
        else:
            return f"Error in LLM request: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# Load the JSON data
with open('/Users/I212229/Desktop/Projects/llm-eval-patriothacks/generation/RFI_Cylinder_Mould_Watermark_Paper_gen_req.json', 'r') as file:
    data = json.load(file)

# Process each requirement and evaluate compatibility
results = []
requirements_summary = ""
for requirement in data["full_pydanctic_output"]["requirements"]:
    requirement_text = requirement["text"]
    pattern_type = evaluate_ears_compatibility(requirement_text)
    
    if pattern_type == "Non-EARS":
        suggested_pattern = suggest_ears_pattern(requirement_text)
    else:
        suggested_pattern = "N/A"
    
    results.append({
        "name": requirement["name"],
        "text": requirement_text,
        "EARS_Type": pattern_type,
        "Suggested_EARS_Pattern": suggested_pattern
    })
    
    # Collect summary for LLM analysis
    requirements_summary += f"Requirement: {requirement['name']}, Type: {pattern_type}, Suggested Pattern: {suggested_pattern}\n"

# Display the results
for result in results:
    print(f"Requirement Name: {result['name']}")
    print(f"Requirement Text: {result['text']}")
    print(f"EARS Type: {result['EARS_Type']}")
    print(f"Suggested EARS Pattern: {result['Suggested_EARS_Pattern']}")
    print("-" * 40)

# LLM Overall Explanation
overall_explanation = overall_llm_explanation(requirements_summary)
print("LLM Overall Explanation:")
print(overall_explanation)

# Calculate EARS Compatibility Score
ears_count = sum(1 for result in results if result["EARS_Type"] != "Non-EARS")
total_requirements = len(results)
compatibility_score = (ears_count / total_requirements) * 100
print(f"EARS Compatibility Score: {compatibility_score:.2f}%")
