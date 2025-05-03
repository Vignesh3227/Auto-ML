import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import time
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, pull as cls_pull, save_model as cls_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from typing import Annotated, Dict, TypedDict, List, Literal, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
import json
from langgraph.graph import StateGraph, END
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
import time  
load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(api_key=mistral_api_key)

class AnalysisState(TypedDict):
    df: Annotated[Optional[pd.DataFrame], "df"]
    target_column: Annotated[Optional[str], "target_column"]
    problem_type: Annotated[Optional[Literal["classification", "regression", "unknown"]], "problem_type"]
    messages: Annotated[List[Dict], "messages"] 
    next_step: Annotated[Optional[str], "next_step"]
    model_results: Annotated[Optional[Dict], "model_results"]
    error: Annotated[Optional[str], "error"]

def analyze_data(state: AnalysisState) -> Dict:
    try:
        df = state["df"]
        target_column = state["target_column"]
        
        if df is None or target_column is None:
            return {
                "error": "Missing dataset or target column", 
                "next_step": "error"
            }
        
        unique_vals = df[target_column].nunique()
        dtype = df[target_column].dtype
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data science expert. Analyze the given dataset statistics 
            and determine if this is a classification or regression problem.
            Consider the data type, number of unique values, and distribution.
            Respond with ONLY 'classification' or 'regression'."""),
            ("human", f"""
            Target column: {target_column}
            Data type: {dtype}
            Number of unique values: {unique_vals}
            Sample values: {df[target_column].sample(min(5, len(df))).tolist()}
            """)
        ])
        
        llm_response = llm.invoke(analysis_prompt.format_messages())
        llm_recommendation = llm_response.content.strip().lower()
        
        if pd.api.types.is_numeric_dtype(dtype) and unique_vals > 10:
            heuristic_type = "regression"
        else:
            heuristic_type = "classification"
            
        if "classification" in llm_recommendation:
            problem_type = "classification"
        elif "regression" in llm_recommendation:
            problem_type = "regression"
        else:
            problem_type = heuristic_type
            
        new_message = {"role": "system", "content": f"Analyzed dataset and determined problem type: {problem_type}"}
        
        return {
            "problem_type": problem_type,
            "messages": [new_message],
            "next_step": "train_model"
        }
    except Exception as e:
        return {
            "error": str(e), 
            "next_step": "error"
        }

def train_model(state: AnalysisState) -> Dict:
    try:
        df = state["df"]
        target_column = state["target_column"]
        problem_type = state["problem_type"]
        
        if df is None or target_column is None or problem_type is None:
            return {
                "error": "Missing required data", 
                "next_step": "error"
            }
        
        if problem_type == "classification":
            cls_setup(df, target=target_column, session_id=123, verbose=False)
            best = cls_compare()
            results = cls_pull()
            cls_save(best, "best_classification_model")
        else:
            reg_setup(df, target=target_column, session_id=123, verbose=False)
            best = reg_compare()
            results = reg_pull()
            reg_save(best, "best_regression_model")
        
        results_dict = results.to_dict()
        
        new_message = {"role": "system", "content": f"Successfully trained {problem_type} models"}
        
        return {
            "model_results": results_dict,
            "messages": [new_message],
            "next_step": "complete"
        }
    except Exception as e:
        return {
            "error": str(e), 
            "next_step": "error"
        }

def handle_error(state: AnalysisState) -> Dict:
    error_msg = state.get("error", "Unknown error occurred")
    new_message = {"role": "system", "content": f"Error: {error_msg}"}
    
    return {
        "messages": [new_message],
        "next_step": END
    }

def build_automl_graph():
    workflow = StateGraph(AnalysisState)
    workflow.add_node("analyze_data", analyze_data)
    workflow.add_node("train_model", train_model)
    workflow.add_node("handle_error", handle_error)
    
    workflow.add_conditional_edges(
        "analyze_data",
        lambda state: "error" if state.get("error") else "train_model",
        {
            "train_model": "train_model",
            "error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "train_model",
        lambda state: "handle_error" if state.get("error") else END,
        {
            END: END,
            "handle_error": "handle_error"
        }
    )

    workflow.add_edge("handle_error", END)

    workflow.set_entry_point("analyze_data")
    
    return workflow.compile()

st.set_page_config(page_title="AutoML Agent with LangGraph", layout="wide")
st.markdown("""
<style>
    /* Hide debug messages */
    div[data-testid="stMarkdownContainer"] p:has(> div.ai-debug) {
        display: none;
    }
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.title("AutoML Agent")
    choice = st.radio("Navigation", ["Upload", "Chat with Data", "Profile Data", "Auto Modeling", "Download Model"])

if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'agent' not in st.session_state:
    st.session_state.agent = None

automl_workflow = build_automl_graph()

if choice == "Upload":
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        with st.spinner("Processing dataset..."):
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.df.to_csv("dataset.csv", index=False)
            st.success("Dataset uploaded successfully!")
            
            st.subheader("Dataset Preview")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
            
            st.subheader("Dataset Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", st.session_state.df.shape[0])
                st.metric("Columns", st.session_state.df.shape[1])
            with col2:
                st.metric("Missing Values", st.session_state.df.isna().sum().sum())
                st.metric("Numeric Columns", len(st.session_state.df.select_dtypes(include=['number']).columns))

elif choice == "Chat with Data":
    st.header("Talk to Your Dataset")
    if "df" not in st.session_state:
        st.session_state.df = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if st.session_state.df is None:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.df.to_csv("dataset.csv", index=False)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.warning("Please upload a dataset first.")
    else:
        if st.session_state.agent is None:
            with st.spinner("Initializing AI agent..."):
                try:
                    st.session_state.agent = create_pandas_dataframe_agent(
                        llm, 
                        st.session_state.df, 
                        verbose=True, 
                        handle_parsing_errors=True, 
                        allow_dangerous_code=True,
                        max_iterations=6,  
                        return_intermediate_steps=True  
                    )
                except Exception as e:
                    st.error(f"Error initializing agent: {str(e)}")
                    st.session_state.agent = None
        
        with st.expander("Dataset Preview", expanded=False):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
            
            st.write("Dataset Information:")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Rows: {st.session_state.df.shape[0]}")
                st.write(f"Columns: {st.session_state.df.shape[1]}")
            with col2:
                numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
                st.write(f"Numeric columns: {len(numeric_cols)}")
                st.write(f"Missing values: {st.session_state.df.isna().sum().sum()}")
        
        st.subheader("Ask questions about your data")
        
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f"""
                <div style="display: flex; margin-bottom: 1rem;">
                    <div style="background-color: #e6f7ff; padding: 1rem; border-radius: 0.5rem; 
                                max-width: 80%; margin-left: auto; color: #2c3e50; 
                                font-size: 16px; line-height: 1.6;">
                        {message}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif role == "ai":
                st.markdown(f"""
                <div style="display: flex; margin-bottom: 1rem;">
                    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; 
                                max-width: 80%; color: #2c3e50; font-size: 16px; 
                                line-height: 1.6; border: 1px solid #dee2e6;">
                        {message}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with st.container():
            user_input = st.text_area("Your question:", placeholder="Ask anything about your data...", height=100)
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                send_button = st.button("Send", use_container_width=True)
            with col2:
                clear_button = st.button("Clear History", use_container_width=True)
            with col3:
                new_data_button = st.button("Upload New Data", use_container_width=True)
                
        if clear_button:
            st.session_state.chat_history = []
            st.experimental_rerun()
        
        if new_data_button:
            st.session_state.df = None
            st.session_state.agent = None
            st.session_state.chat_history = []
            st.experimental_rerun()
                
        if (send_button or st.session_state.get("process_input", False)) and user_input:
            if not st.session_state.get("process_input", False):
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.current_question = user_input
                st.session_state.process_input = True
                st.experimental_rerun()
            
            user_input = st.session_state.current_question
            with st.spinner("Analyzing your data..."):
                try:
                    def extract_answer_from_error(error_message):
                        if "Final Answer:" in error_message:
                            answer_parts = error_message.split("Final Answer:")
                            if len(answer_parts) > 1:
                                return answer_parts[1].strip()

                        if "Observation:" in error_message:
                            observation_parts = error_message.split("Observation:")
                            if len(observation_parts) > 1:
                                return f"Based on the data analysis: {observation_parts[1].strip()}"
                        return None
                    
                    try:
                        raw_response = st.session_state.agent.invoke({"input": user_input})
                        raw_output = raw_response.get("output", str(raw_response))
                    except Exception as agent_error:

                        error_str = str(agent_error)
                        extracted_answer = extract_answer_from_error(error_str)
                        if extracted_answer:
                            raw_output = extracted_answer
                        else:

                            raise

                    summary_prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are a data analyst. Convert this technical response into a clear, 
                        concise summary using plain English. Follow these rules:
                        1. Use simple language a business user would understand
                        2. Highlight key numbers and trends
                        3. Present important insights clearly
                        4. Include actionable conclusions when possible
                        5. Never mention Python/pandas code
                        6. Format with bullet points for clarity when appropriate
                         7. Ignore errors and links"""),
                        ("human", "Technical response: {raw_output}")
                    ])
                    
                    formatted_prompt = summary_prompt.format_messages(raw_output=raw_output)
                    summary_response = llm.invoke(formatted_prompt)
                    summary_text = summary_response.content

                    st.session_state.chat_history.append(("ai", summary_text))
                    
                except Exception as e:
                    error_message = f"I encountered an error while analyzing your data: {str(e)}"
                    st.session_state.chat_history.append(("ai", error_message))
                st.session_state.process_input = False
                st.session_state.current_question = ""
            
            st.experimental_rerun()

elif choice == "Profile Data":
    st.header("Profile Your Dataset")
    
    if st.session_state.df is None:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            with st.spinner("Loading dataset..."):
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.df.to_csv("dataset.csv", index=False)
                st.experimental_rerun()
        else:
            st.warning("Please upload a dataset first.")
    else:
        with st.expander("Dataset Preview", expanded=False):
            st.dataframe(st.session_state.df.head(5), use_container_width=True)
        
        st.subheader("Generate Profiling Report")
        
        col1, col2 = st.columns(2)
        with col1:
            minimal = st.checkbox("Minimal Report (Faster)", value=False)
        with col2:
            sample_size = st.slider("Sample Size (%)", min_value=10, max_value=100, value=100, step=10)
        
        if st.button("Generate Profile Report"):
            with st.spinner("Generating comprehensive profile report... This may take a few minutes."):
                try:
                    if sample_size < 100:
                        df_sample = st.session_state.df.sample(frac=sample_size/100, random_state=42)
                    else:
                        df_sample = st.session_state.df
                    
                    if minimal:
                        profile = ProfileReport(df_sample, minimal=True, title="Dataset Profile Report")
                    else:
                        profile = ProfileReport(df_sample, title="Dataset Profile Report")
                    
                    st_profile_report(profile)
                    
                    profile_html = profile.to_html()
                    st.download_button(
                        label="Download Full HTML Report",
                        data=profile_html,
                        file_name="profile_report.html",
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"Error generating profile report: {str(e)}")
                    st.info("Try using a smaller sample size or the minimal report option for large datasets.")

elif choice == "Auto Modeling":
    st.header("Automatic Model Selection and Training")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        target_column = st.selectbox("Select Target Column", st.session_state.df.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            auto_detect = st.toggle("Auto-detect problem type", value=True)
        
        with col2:
            if not auto_detect:
                problem_type = st.selectbox("Select problem type", ["classification", "regression"])
            else:
                problem_type = None
        
        if st.button("Analyze & Train Models"):
            with st.spinner("Running LangGraph workflow for intelligent model selection..."):
                initial_state = {
                    "df": st.session_state.df,
                    "target_column": target_column,
                    "problem_type": problem_type,
                    "messages": [],
                    "next_step": None,
                    "model_results": None,
                    "error": None
                }
                
                result = automl_workflow.invoke(initial_state)
                
                if result.get("error"):
                    st.error(f"Error during workflow: {result['error']}")
                else:
                    problem_type = result["problem_type"]
                    st.session_state.model_results = result["model_results"]
                    
                    st.success(f"Detected problem type: **{problem_type.capitalize()}**")
                    st.success("Models trained successfully!")
                    
                    if result["model_results"]:
                        results_df = pd.DataFrame(result["model_results"])
                        st.subheader("Model Comparison Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        best_model = results_df.iloc[0].name if not results_df.empty else "Unknown"
                        st.info(f"Best model: **{best_model}**")
                        
                        st.session_state.best_model_name = f"best_{problem_type}_model.pkl"

elif choice == "Download Model":
    st.header("Download Your Trained Model")
    
    if 'best_model_name' not in st.session_state or not os.path.exists(st.session_state.best_model_name):
        st.warning("No trained models found. Please run Auto Modeling first.")
    else:
        st.subheader("Best Trained Model")
        model_file = st.session_state.best_model_name
        model_type = "classification" if "classification" in model_file else "regression"
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            **Model Details:**
            - File name: {model_file}
            - Type: {model_type.capitalize()}
            - Last modified: {time.ctime(os.path.getmtime(model_file))}
            """)
            
        with col2:
            with open(model_file, "rb") as f:
                st.download_button(
                    label="Download Best Model",
                    data=f,
                    file_name=model_file,
                    mime="application/octet-stream",
                    key="download_best_model"
                )
        
        with st.expander("How to use your model"):
            st.markdown("""
            ### Loading your saved model
            
            ```python
            # For classification models
            from pycaret.classification import load_model
            model = load_model('best_classification_model')
            
            # For regression models
            from pycaret.regression import load_model
            model = load_model('best_regression_model')
            
            # Make predictions
            predictions = model.predict(new_data)
            ```
            """)
st.markdown("---")
st.markdown("AutoML Agent with LangGraph - Powered by PyCaret and MistralAI")