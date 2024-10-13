import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import json

# Load spacy NER model
nlp = spacy.load("en_core_web_sm")

# Function to extract important keywords (nouns, verbs, proper nouns)
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'VERB']]
    return set(keywords)

# Function to calculate cosine similarity between two sets of keywords
def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

# Function to evaluate faithfulness using both NER and keyword overlap
def evaluate_faithfulness(requirements):
    results = []
    for req in requirements:
        source_text = req["source"]["source text"]
        generated_requirement = req["text"]

        # Extract NER entities and keywords
        source_keywords = extract_keywords(source_text)
        generated_keywords = extract_keywords(generated_requirement)

        # Check for empty sets
        if not source_keywords and not generated_keywords:
            overlap_score = 0.0
            llm_explanation = "No significant keywords were found in either the source or the generated requirement."
        else:
            overlap_score = len(source_keywords.intersection(generated_keywords)) / len(source_keywords.union(generated_keywords))

        results.append({
            "requirement": generated_requirement,
            "source": source_text,
            "faithful": overlap_score >= 0.5,
            "similarity_score": overlap_score
        })
    
    return results

# Function to generate LLM explanation
def generate_llm_explanation(source_keywords, generated_keywords, score):
    if score < 0.5:
        return f"The generated requirement deviates significantly from the source. For instance, important terms like {list(source_keywords.difference(generated_keywords))[:2]} are missing or misrepresented."
    else:
        return f"The generated requirement is mostly aligned with the source. Most terms, such as {list(source_keywords.intersection(generated_keywords))[:2]}, match correctly."

# Example usage (will only run if this script is executed directly)
if __name__ == "__main__":
    # Load the JSON data
    with open("RFI_Cylinder_Mould_Watermark_Paper_gen_req.json", "r") as file:
        data = json.load(file)

    requirements = data["full_pydanctic_output"]["requirements"]
    
    # Run evaluation
    faithfulness_results = evaluate_faithfulness(requirements)
    for result in faithfulness_results:
        print(f"Requirement: {result['requirement']}")
        print(f"Source Text: {result['source']}")
        print(f"Faithful: {result['faithful']}")
        print(f"Similarity Score: {result['similarity_score']:.2f}")
        print("-" * 40)