import streamlit as st
import os

st.set_page_config(page_title="Requirements Evaluation Dashboard", layout="wide")

st.title("Requirements Evaluation Dashboard")

# Function to run all evaluations
def run_evaluations(pdf_path):
    # Import necessary modules only when function is called
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from coherence import extract_requirements, evaluate_coherence
    from faithfulness_eval import detect_faithfulness_extraction
    from hallucination_eval import detect_hallucination_with_explanation
    from ears_score import calculate_ears_score
    from score_eval import evaluate_content_coverage
    import re

    requirements_data = extract_requirements(pdf_path)
    
    # Coherence Evaluation
    requirements_text = " ".join([req['text'] for req in requirements_data['full_pydanctic_output']['requirements']])
    coherence_result = evaluate_coherence(requirements_text)
    
    # Faithfulness Evaluation
    faithfulness_results = detect_faithfulness_extraction(pdf_path)
    
    # Hallucination Detection
    hallucination_results = detect_hallucination_with_explanation(pdf_path)
    
    # EARS Score Evaluation
    ears_result = calculate_ears_score(requirements_data['full_pydanctic_output']['requirements'])
    
    # Content Coverage Evaluation
    coverage_results = evaluate_content_coverage(pdf_path)
    
    return {
        'requirements_data': requirements_data,
        'coherence_result': coherence_result,
        'faithfulness_results': faithfulness_results,
        'hallucination_results': hallucination_results,
        'ears_result': ears_result,
        'coverage_results': coverage_results
    }

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    st.success("File successfully uploaded!")
    
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Add a button to trigger evaluation
    if st.button("Run Evaluation"):
        # Run evaluations
        with st.spinner('Running evaluations...'):
            try:
                results = run_evaluations("temp.pdf")
                
                # Import visualization libraries only when needed
                import pandas as pd
                import plotly.graph_objects as go
                import plotly.express as px
                import re
                
                # Create three columns for the dashboard
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Coherence Evaluation
                    st.header("Coherence Evaluation")
                    coherence_score = float(re.search(r'(\d+(\.\d+)?)', results['coherence_result']).group(1))
                    
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
                        st.write(results['coherence_result'])
                
                with col2:
                    # Faithfulness Evaluation
                    st.header("Faithfulness Evaluation")
                    avg_faithfulness = sum(result['faithful'] for result in results['faithfulness_results']) / len(results['faithfulness_results']) * 100
                    
                    st.markdown(f"<h1 style='text-align: center; color: green;'>{avg_faithfulness:.1f}%</h1>", unsafe_allow_html=True)
                    
                    df = pd.DataFrame(results['faithfulness_results'])
                    df['faithful_emoji'] = df['faithful'].apply(lambda x: '✅' if x else '❌')
                    
                    st.dataframe(df[['requirement', 'faithful_emoji', 'similarity_score']], height=300)
                
                with col3:
                    # Hallucination Detection
                    st.header("Hallucination Detection")
                    st.write(f"Hallucination Detected: {'Yes' if results['hallucination_results']['hallucination_detected'] else 'No'}")
                    
                    hallucination_df = pd.DataFrame(results['hallucination_results']['explanations'])
                    hallucination_df['hallucination_emoji'] = hallucination_df['hallucination'].apply(lambda x: '❌' if x else '✅')
                    
                    st.dataframe(hallucination_df[['iteration', 'hallucination_emoji', 'similarity_score']], height=300)
                    
                    with st.expander("Hallucination Details"):
                        for exp in results['hallucination_results']['explanations']:
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
                    fig = px.pie(
                        values=[results['ears_result']['ears_count'], results['ears_result']['non_ears_count']],
                        names=['EARS', 'Non-EARS'],
                        title=f"EARS Compatibility: {results['ears_result']['ears_score']:.1f}%"
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    ears_df = pd.DataFrame(results['ears_result']['results'])
                    ears_df['ears_emoji'] = ears_df['is_ears'].apply(lambda x: '✅' if x else '❌')
                    
                    st.dataframe(ears_df[['requirement', 'ears_emoji', 'ears_type']], height=300)
                    
                    with st.expander("EARS Explanation"):
                        st.write(results['ears_result']['explanation'])
                
                with col5:
                    # Content Coverage Evaluation
                    st.header("Content Coverage Evaluation")
                    if results['coverage_results']:
                        coverage_df = pd.DataFrame(results['coverage_results'])
                        
                        average_scores = coverage_df[['rouge1_recall', 'rouge2_recall', 'rougeL_recall']].mean()
                        
                        fig = go.Figure([go.Bar(x=average_scores.index, y=average_scores.values)])
                        fig.update_layout(title="Average ROUGE Scores", xaxis_title="ROUGE Metric", yaxis_title="Score")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(coverage_df, height=300)
                    else:
                        st.warning("No content coverage results available.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
            finally:
                # Clean up
                os.remove("temp.pdf")
    else:
        st.info("Click 'Run Evaluation' to start the analysis.")
else:
    st.info("Please upload a PDF document to start the evaluation.")