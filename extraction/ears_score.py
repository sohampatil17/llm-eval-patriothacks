import re
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# EARS regex patterns
ears_patterns = {
    "Ubiquitous": re.compile(r".*shall.*"),
    "State_Driven": re.compile(r"While .*shall.*"),
    "Event_Driven": re.compile(r"When .*shall.*"),
    "Optional_Feature": re.compile(r"Where .*shall.*"),
    "Unwanted_Behavior": re.compile(r"If .*then .*shall.*"),
    "Complex": re.compile(r"While .*When .*shall.*")
}

def evaluate_ears_compatibility(requirement_text):
    for pattern_name, pattern in ears_patterns.items():
        if pattern.match(requirement_text):
            return pattern_name
    return "Non-EARS"

def get_ears_recommendation(requirement_text):
    prompt = f"Convert the following requirement into an EARS (Easy Approach to Requirements Syntax) compliant format:\n\n{requirement_text}\n\nEARS-compliant version:"
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in EARS (Easy Approach to Requirements Syntax) for writing requirements."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating EARS recommendation: {str(e)}"

def calculate_ears_score(requirements):
    results = []
    ears_count = 0
    total_requirements = len(requirements)

    for requirement in requirements:
        req_text = requirement.get('text', '')
        pattern_type = evaluate_ears_compatibility(req_text)
        
        if pattern_type != "Non-EARS":
            ears_count += 1
            ears_recommendation = "N/A"
        else:
            ears_recommendation = get_ears_recommendation(req_text)
        
        results.append({
            "text": req_text,
            "ears_type": pattern_type,
            "ears_recommendation": ears_recommendation
        })

    compatibility_score = (ears_count / total_requirements) * 100 if total_requirements > 0 else 0

    return {
        'ears_score': compatibility_score,
        'results': results,
        'ears_count': ears_count,
        'non_ears_count': total_requirements - ears_count
    }

# Example usage (commented out to prevent execution on import)
# if __name__ == "__main__":
#     sample_requirements = [
#         {"text": "The system shall provide user authentication."},
#         {"text": "Display an error message if the login fails."}
#     ]
#     result = calculate_ears_score(sample_requirements)
#     print(json.dumps(result, indent=2))