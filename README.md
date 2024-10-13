# LLM.evals 🌲💯
### AI-Powered Requirements Engineering Evaluation

<img width="1754" alt="Screenshot 2024-10-13 at 9 32 42 AM" src="https://github.com/user-attachments/assets/f047bccb-64ad-4cf0-8a69-a1845be78e18">

<br />
## 🎯 Project Overview

LLM.evals is a cutting-edge tool designed to evaluate the quality and relevance of requirements generated or extracted by Large Language Models (LLMs) in the context of Systems Engineering. Our project addresses the critical need for explainable and measurable AI outputs in the defense industry, focusing on requirements engineering.

## 🔬 Core Evaluation Metrics

1. **Coherence Analysis** 📊
   - Utilizes GPT-4 for semantic evaluation
   - Scores requirements on a 1-5 scale for overall flow and relatedness
   - Coherence score calculation:
     ```
     C = (1/N) * Σ(w_i * s_i)
     ```
     where C is the coherence score, N is the number of requirements, w_i is the weight of the i-th requirement, and s_i is the individual coherence score of the i-th requirement.

2. **Faithfulness Assessment** 🔍
   - Implements spaCy for Natural Language Processing
   - Extracts key linguistic elements (nouns, verbs, proper nouns)
   - Calculates Jaccard similarity for overlap score:
     ```
     J(A,B) = |A ∩ B| / |A ∪ B|
     ```
     where A is the set of key elements from the source text and B is the set from the generated requirement.

3. **Hallucination Detection** 🕵️
   - Employs BERT embeddings for semantic similarity computation
   - Applies cosine similarity metrics to identify potential fabrications:
     ```
     similarity = cos(θ) = (A · B) / (||A|| ||B||)
     ```
     where A and B are the BERT embeddings of the source and generated texts.
   - Uses SequenceMatcher for granular text difference analysis

4. **EARS (Easy Approach to Requirements Syntax) Compatibility** 📏
   - Implements regex pattern matching for EARS classification
   - Categories: Ubiquitous, State-Driven, Event-Driven, Optional Feature, Unwanted Behavior, Complex
   - EARS Compatibility Score:
     ```
     EARS_score = (N_compatible / N_total) * 100
     ```
     where N_compatible is the number of EARS-compatible requirements and N_total is the total number of requirements.

5. **Content Coverage Evaluation** 📄 (Extraction mode only)
   - Utilizes ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores
   - Calculates ROUGE-1, ROUGE-2, and ROUGE-L:
     ```
     ROUGE-N = Σ(gram_n ∈ RefSum) Count_match(gram_n) / Σ(gram_n ∈ RefSum) Count(gram_n)
     ```
     where gram_n is an n-gram, RefSum is the reference summary, and Count_match is the maximum number of n-grams co-occurring in the candidate summary and reference summary.

## 💻 Technical Implementation

### Data Processing Pipeline

1. **Document Ingestion**
   - Supports PDF (Extraction) and JSON (Generation) inputs
   - Implements custom API for requirement extraction from PDFs
   - PDF processing using PyPDF2 or pdfminer.six libraries

2. **Text Preprocessing**
   - Utilizes TfidfVectorizer for text vectorization:
     ```python
     from sklearn.feature_extraction.text import TfidfVectorizer
     vectorizer = TfidfVectorizer()
     tfidf_matrix = vectorizer.fit_transform(documents)
     ```
   - Applies custom tokenization and normalization techniques
   - Implements stop word removal and lemmatization using NLTK or spaCy

3. **LLM Integration**
   - Interfaces with GPT-4 and custom-trained LLMs
   - Implements prompt engineering for optimal requirement generation
   - Uses OpenAI API for GPT-4 integration:
     ```python
     import openai
     openai.api_key = 'your-api-key'
     response = openai.Completion.create(
         engine="text-davinci-002",
         prompt=engineered_prompt,
         max_tokens=150
     )
     ```

### Evaluation Modules

1. **Coherence Module**
   - Leverages GPT-4's contextual understanding
   - Implements custom scoring algorithm based on LLM outputs
   - Utilizes sliding window approach for long-range coherence:
     ```python
     def sliding_window_coherence(text, window_size=3):
         sentences = nltk.sent_tokenize(text)
         scores = []
         for i in range(len(sentences) - window_size + 1):
             window = sentences[i:i+window_size]
             score = evaluate_coherence(window)
             scores.append(score)
         return sum(scores) / len(scores)
     ```

2. **Faithfulness Module**
   - Utilizes spaCy's linguistic feature extraction
   - Implements custom set theory algorithms for overlap calculation
   - Example of key element extraction:
     ```python
     import spacy
     nlp = spacy.load("en_core_web_sm")
     def extract_key_elements(text):
         doc = nlp(text)
         return set([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'VERB', 'PROPN']])
     ```

3. **Hallucination Detection Module**
   - Integrates HuggingFace's BERT model for embedding generation:
     ```python
     from transformers import BertTokenizer, BertModel
     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
     model = BertModel.from_pretrained('bert-base-uncased')
     ```
   - Implements cosine similarity calculation using sklearn:
     ```python
     from sklearn.metrics.pairwise import cosine_similarity
     similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
     ```

4. **EARS Compatibility Module**
   - Uses regex for pattern matching and classification
   - Example of EARS pattern matching:
     ```python
     import re
     ears_patterns = {
         "Ubiquitous": r"The .* shall .*",
         "Event_Driven": r"When .*, the .* shall .*",
         # ... other patterns
     }
     def classify_ears(requirement):
         for pattern_name, pattern in ears_patterns.items():
             if re.match(pattern, requirement):
                 return pattern_name
         return "Non-EARS"
     ```

5. **Content Coverage Module**
   - Implements ROUGE metric calculations
   - Utilizes custom algorithms for precision and recall computations
   - Example of ROUGE-1 calculation:
     ```python
     from rouge import Rouge
     rouge = Rouge()
     scores = rouge.get_scores(generated_text, reference_text)
     rouge1_score = scores[0]['rouge-1']
     ```

## 🚀 Future Enhancements

1. **Advanced NLP Techniques**
   - Implement transformer-based models for deeper semantic analysis
   - Explore few-shot learning for improved requirement classification

2. **Expanded Input Support**
   - Develop modules to support additional document formats (e.g., Word, Markdown)
   - Implement OCR capabilities for handling scanned documents

3. **LLM Comparison Framework**
   - Develop a system to compare outputs from different LLMs
   - Implement statistical analysis for performance benchmarking

4. **Integration with SE Tools**
   - Create APIs for seamless integration with popular Systems Engineering tools
   - Develop plugins for MBSE (Model-Based Systems Engineering) platforms

5. **Enhanced EARS Recommendation System**
   - Implement machine learning models for context-aware EARS suggestions
   - Develop a feedback loop system for continuous improvement of recommendations

By focusing on these technical aspects and mathematical foundations, LLM.evals 🌲💯 aims to revolutionize the requirements engineering process in the defense industry, providing a robust, explainable, and measurable approach to AI-assisted systems engineering.
