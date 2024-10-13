import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to safely import modules
def safe_import(module_name, function_names):
    try:
        module = __import__(module_name, fromlist=function_names)
        return {name: getattr(module, name) for name in function_names}
    except ImportError as e:
        st.error(f"Error importing {module_name}: {str(e)}")
        return {name: lambda *args, **kwargs: None for name in function_names}
    except AttributeError as e:
        st.error(f"Error importing function from {module_name}: {str(e)}")
        return {name: lambda *args, **kwargs: None for name in function_names}

# Importing extraction evaluation functions
extraction_functions = safe_import("extraction.coherence", ["extract_requirements", "evaluate_coherence"])
extraction_functions.update(safe_import("extraction.faithfulness_eval", ["detect_faithfulness_extraction"]))
extraction_functions.update(safe_import("extraction.hallucination_eval", ["detect_hallucination_with_explanation"]))
extraction_functions.update(safe_import("extraction.ears_score", ["calculate_ears_score"]))
extraction_functions.update(safe_import("extraction.score_eval", ["evaluate_content_coverage"]))

# Importing generation evaluation functions
generation_functions = safe_import("generation.coherence", ["evaluate_coherence"])
generation_functions.update(safe_import("generation.faithfulness_eval", ["evaluate_faithfulness"]))
generation_functions.update(safe_import("generation.hallucination_eval", ["detect_hallucination"]))
generation_functions.update(safe_import("generation.ears_score", ["evaluate_ears_score"]))

# Assign imported functions to variables
extract_requirements = extraction_functions["extract_requirements"]
evaluate_coherence = extraction_functions["evaluate_coherence"]
detect_faithfulness_extraction = extraction_functions["detect_faithfulness_extraction"]
detect_hallucination_with_explanation = extraction_functions["detect_hallucination_with_explanation"]
calculate_ears_score = extraction_functions["calculate_ears_score"]
evaluate_content_coverage = extraction_functions["evaluate_content_coverage"]

evaluate_coherence_generation = generation_functions["evaluate_coherence"]
evaluate_faithfulness = generation_functions["evaluate_faithfulness"]
detect_hallucination = generation_functions["detect_hallucination"]
evaluate_ears_score = generation_functions["evaluate_ears_score"]

st.set_page_config(page_title="LLM.evals", layout="wide")

st.title("LLM.evals 🌲💯")

# Mode selection
mode = st.radio("Select mode:", ("Extraction", "Generation"))

if mode == "Extraction":
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        st.success("File successfully uploaded!")
        
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Run evaluations
        with st.spinner('Running evaluations...'):
            try:
                # Extract requirements
                requirements_data = extract_requirements("temp.pdf")
                
                # Create three columns for the first row
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Coherence Evaluation
                    st.header("Coherence Evaluation")
                    requirements_text = " ".join([req['text'] for req in requirements_data['full_pydanctic_output']['requirements']])
                    coherence_result = evaluate_coherence(requirements_text)
                    
                    # Extract numeric score from the result
                    coherence_score = float(re.search(r'(\d+(\.\d+)?)', coherence_result).group(1))
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = coherence_score,
                        title = {'text': "Coherence Score"},
                        gauge = {
                            'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 2], 'color': 'red'},
                                {'range': [2, 3.5], 'color': 'yellow'},
                                {'range': [3.5, 5], 'color': 'green'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 4
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Coherence Explanation"):
                        st.write(coherence_result)
                
                with col2:
                    # Faithfulness Evaluation
                    st.header("Faithfulness Evaluation")
                    faithfulness_results = detect_faithfulness_extraction("temp.pdf")
                    
                    # Calculate average faithfulness score
                    avg_faithfulness = sum(result['faithful'] for result in faithfulness_results) / len(faithfulness_results) * 100
                    
                    st.markdown(f"<h1 style='text-align: center; color: green;'>{avg_faithfulness:.1f}%</h1>", unsafe_allow_html=True)
                    
                    # Create a DataFrame with emojis for faithful/not faithful
                    df = pd.DataFrame(faithfulness_results)
                    df['index'] = df['faithful'].apply(lambda x: '✅' if x else '❌')
                    df.set_index('index', inplace=True)
                    
                    # Display scrollable table
                    st.dataframe(df[['requirement', 'similarity_score']], height=300)
                
                with col3:
                    # Hallucination Detection
                    st.header("Hallucination Detection")
                    hallucination_results = detect_hallucination_with_explanation("temp.pdf")
                    
                    hallucination_detected = hallucination_results['hallucination_detected']
                    
                    if hallucination_detected:
                        st.markdown("<h2 style='text-align: center; color: red;'>Hallucination Detected</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h2 style='text-align: center; color: green;'>No Hallucination Detected</h2>", unsafe_allow_html=True)
                    
                    st.subheader("Technical Information")
                    st.write("Model: BERT-based semantic similarity")
                    st.write("Iterations: 3")
                    st.write("Similarity Threshold: 0.85")
                    
                    # Create a DataFrame for hallucination results
                    hallucination_df = pd.DataFrame(hallucination_results['explanations'])
                    hallucination_df['index'] = hallucination_df['hallucination'].apply(lambda x: '❌' if x else '✅')
                    hallucination_df.set_index('index', inplace=True)
                    
                    # Display scrollable table
                    st.dataframe(hallucination_df[['iteration', 'similarity_score']], height=200)
                    
                    # Display details in expander
                    with st.expander("Hallucination Details"):
                        for exp in hallucination_results['explanations']:
                            st.subheader(f"Iteration {exp['iteration']}")
                            st.write(f"Similarity Score: {exp['similarity_score']:.2f}")
                            st.write("Differences:")
                            for diff in exp['differences']:
                                st.write(f"- Change: {diff['change']}")
                                st.write(f"  From: {diff['from']}")
                                st.write(f"  To: {diff['to']}")
                
                # Create two columns for the second row
                col4, col5 = st.columns(2)
                
                with col4:
                    # EARS Score Evaluation
                    st.header("EARS Score Evaluation")
                    ears_result = calculate_ears_score(requirements_data['full_pydanctic_output']['requirements'])
                    
                    fig = px.pie(
                        values=[ears_result['ears_count'], ears_result['non_ears_count']],
                        names=['EARS', 'Non-EARS'],
                        title=f"EARS Compatibility: {ears_result['ears_score']:.1f}%",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    ears_df = pd.DataFrame(ears_result['results'])
                    ears_df['index'] = ears_df['ears_type'].apply(lambda x: '✅' if x != 'Non-EARS' else '❌')
                    ears_df.set_index('index', inplace=True)
                    
                    st.dataframe(ears_df[['text', 'ears_type', 'ears_recommendation']], height=300)
                        
                with col5:
                    # Content Coverage Evaluation
                    st.header("Content Coverage Evaluation")
                    try:
                        with st.spinner('Evaluating content coverage...'):
                            coverage_results = evaluate_content_coverage("temp.pdf")
                        if coverage_results and isinstance(coverage_results, list) and len(coverage_results) > 0:
                            coverage_df = pd.DataFrame(coverage_results)
                            
                            average_scores = coverage_df[['rouge1_recall', 'rouge2_recall', 'rougeL_recall']].mean()
                            
                            fig = go.Figure([go.Bar(
                                x=average_scores.index, 
                                y=average_scores.values,
                                marker_color=px.colors.qualitative.Pastel
                            )])
                            fig.update_layout(title="Average ROUGE Scores", xaxis_title="ROUGE Metric", yaxis_title="Score")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.dataframe(
                                coverage_df,
                                column_config={
                                    "requirement": st.column_config.TextColumn("Requirement", width="medium"),
                                    "source": st.column_config.TextColumn("Source", width="medium"),
                                    "rouge1_recall": st.column_config.NumberColumn("ROUGE-1 Recall", format="%.4f"),
                                    "rouge2_recall": st.column_config.NumberColumn("ROUGE-2 Recall", format="%.4f"),
                                    "rougeL_recall": st.column_config.NumberColumn("ROUGE-L Recall", format="%.4f"),
                                },
                                height=300
                            )
                        else:
                            st.warning("No content coverage results available. The evaluation function returned empty or invalid results.")
                            st.write("Debug information:")
                            st.json(coverage_results)
                    except Exception as e:
                        st.error(f"An error occurred during content coverage evaluation: {str(e)}")
                        st.warning("Please check the evaluate_content_coverage function in your score_eval.py file.")
                        st.write("Debug information:")
                        import traceback
                        st.code(traceback.format_exc())
                        
            finally:
                # Clean up
                os.remove("temp.pdf")

elif mode == "Generation":
    uploaded_file = st.file_uploader("Upload a JSON file", type="json")
    
    if uploaded_file is not None:
        st.success("File successfully uploaded!")
        
        try:
            # Load the JSON data
            json_data = json.load(uploaded_file)
            
            # Run evaluations
            with st.spinner('Running evaluations...'):
                try:
                    # Create two columns for the first row
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Coherence Evaluation
                        st.header("Coherence Evaluation")
                        requirements_text = " ".join([req['text'] for req in json_data['full_pydanctic_output']['requirements']])
                        coherence_result = evaluate_coherence_generation(requirements_text)
                        
                        # Extract numeric score from the result
                        coherence_score = float(re.search(r'(\d+(\.\d+)?)', coherence_result).group(1))
                        
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = coherence_score,
                            title = {'text': "Coherence Score"},
                            gauge = {
                                'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "darkblue"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 2], 'color': 'red'},
                                    {'range': [2, 3.5], 'color': 'yellow'},
                                    {'range': [3.5, 5], 'color': 'green'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 4
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("Coherence Explanation"):
                            st.write(coherence_result)
                    
                    with col2:
                        # Faithfulness Evaluation
                        st.header("Faithfulness Evaluation")
                        faithfulness_results = evaluate_faithfulness(json_data['full_pydanctic_output']['requirements'])
                        
                        # Calculate average faithfulness score
                        avg_faithfulness = sum(result['faithful'] for result in faithfulness_results) / len(faithfulness_results) * 100
                        
                        st.markdown(f"<h1 style='text-align: center; color: green;'>{avg_faithfulness:.1f}%</h1>", unsafe_allow_html=True)
                        
                        # Create a DataFrame with emojis for faithful/not faithful
                        df = pd.DataFrame(faithfulness_results)
                        df['index'] = df['faithful'].apply(lambda x: '✅' if x else '❌')
                        df.set_index('index', inplace=True)
                        
                        # Display scrollable table
                        st.dataframe(df[['requirement', 'similarity_score']], height=300)
                    
                    # Create two columns for the second row
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.header("Hallucination Detection")
                        hallucination_results = detect_hallucination(json_data)
                        
                        hallucination_detected = hallucination_results['hallucination_detected']
                        
                        if hallucination_detected:
                            st.markdown("<h2 style='text-align: center; color: red;'>Hallucination Detected</h2>", unsafe_allow_html=True)
                        else:
                            st.markdown("<h2 style='text-align: center; color: green;'>No Hallucination Detected</h2>", unsafe_allow_html=True)
                        
                        # Create a DataFrame for hallucination results
                        hallucination_df = pd.DataFrame(hallucination_results['explanations'])
                        hallucination_df['index'] = hallucination_df['hallucination'].apply(lambda x: '❌' if x else '✅')
                        hallucination_df.set_index('index', inplace=True)
                        
                        # Display scrollable table
                        st.dataframe(hallucination_df[['iteration', 'similarity_score']], height=200)
                        
                        # Display details in expander
                        with st.expander("Hallucination Details"):
                            for exp in hallucination_results['explanations']:
                                st.subheader(f"Iteration {exp['iteration']}")
                                st.write(f"Similarity Score: {exp['similarity_score']:.2f}")
                                st.write(f"Hallucination Detected: {'Yes' if exp['hallucination'] else 'No'}")
                    
                    with col4:
                        st.header("EARS Score Evaluation")
                        ears_result = evaluate_ears_score(json_data)
                        
                        fig = px.pie(
                            values=[ears_result['ears_count'], ears_result['non_ears_count']],
                            names=['EARS', 'Non-EARS'],
                            title=f"EARS Compatibility: {ears_result['ears_score']:.1f}%",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        ears_df = pd.DataFrame(ears_result['results'])
                        ears_df['index'] = ears_df['is_ears'].apply(lambda x: '✅' if x else '❌')
                        ears_df.set_index('index', inplace=True)
                        
                        st.dataframe(ears_df[['requirement', 'ears_type', 'ears_recommendation']], height=300)
                except Exception as e:
                    st.error(f"An error occurred during generation evaluation: {str(e)}")
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid JSON file.")
        except Exception as e:
            st.error(f"An error occurred while processing the JSON file: {str(e)}")
    else:
        st.info("Please upload a JSON file to start the evaluation.")
else:
    st.info("Please select a mode and upload the appropriate file to start the evaluation.")

# Add some spacing
st.write("")
st.write("")

# Display information about the evaluation process
st.header("About the Evaluation Process")
st.write("""
This dashboard provides a comprehensive evaluation of requirements based on various metrics:

1. **Coherence**: Measures how well the requirements flow and relate to each other.
2. **Faithfulness**: Assesses how closely the generated requirements align with the source text.
3. **Hallucination Detection**: Identifies potential fabrications or inconsistencies in the requirements.
4. **EARS Compatibility**: Evaluates the adherence to the Easy Approach to Requirements Syntax (EARS).
5. **Content Coverage** (Extraction mode only): Measures how well the extracted requirements cover the content of the source document.

The extraction mode processes PDF documents, while the generation mode analyzes JSON files containing generated requirements.
""")

# Add footer
st.markdown("---")
st.markdown("© Built by Soham Patil at PatriotHacks 2024.")
