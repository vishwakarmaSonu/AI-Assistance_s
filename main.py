import os
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph,END
from typing import TypedDict
import streamlit as st



api_key = "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"

# llm for pdf
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

new_vector_store = FAISS.load_local(
    "https://github.com/vishwakarmaSonu/AI-Assistance_s/tree/main/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True   # required in newer LangChain versions
)

retriever = new_vector_store.as_retriever()

llm_2 = ChatGroq(
    api_key=api_key,
    model="gemma2-9b-it",
    temperature=0.0
)

qa_chain = RetrievalQA.from_chain_type(
    llm =llm_2,
    retriever = retriever,
    
)


# csv


llm_3 = ChatGroq(
    api_key=api_key,
    model="gemma2-9b-it",
    temperature=0.0
)

csv_path= "F:\Vishwakarma_Art\AI agent\sales.csv"

pandas_prompt=PromptTemplate(
    input_variables=["input","columns"],
    template="""   
you are a python data analyst. Given a DataFrame called `df` with the following columns:
{columns}
Write a valid one-line pandas expression to answer the question:
"{input}"

only return the python code (no explanation, no backticks).
"""
)

def query_csv(question):
    df = pd.read_csv(csv_path)
    columns =", ".join(df.columns)
    prompt = pandas_prompt.format(input=question, columns=columns)

    response=llm_3.invoke(prompt)

    pandas_code = response.content
    print("Generated code:", pandas_code)

    try:
        result = eval(pandas_code, {"df":df , "pd":pd})

        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result.head().to_string(index=False)
        else:
            return str(result)
    except Exception as e:
        return f"Error {e}"
    

llm_general = ChatGroq(
    api_key=api_key,
    model="gemma2-9b-it",
    temperature=0.7
)
    
# Langgraph

class State(TypedDict):
    query:str
    response:str

def general_node(state: State):
    query = state["query"]
    response = llm_general.invoke(query)
    return {"response": response.content}

def domain_node(state:State):
    query = state['query']
    result = qa_chain(query)
    return {"response":result['result']}
    
def csv_node(state:State):
    query1= state['query']
    return {"response": query_csv(query1)}
    
def router_node(state: State):
    query = state["query"].lower()

    if "csv" in query or "data" in query:
        return {"next": "csv"}
    elif any(greet in query for greet in ["hi", "hello", "hey", "how are you", "who are you"]):
        return {"next": "general"}
    else:
        return {"next": "domain"}  

# Build graph again
workflow = StateGraph(State)

workflow.add_node("router", router_node)
workflow.add_node("general", general_node)
workflow.add_node("domain", domain_node)
workflow.add_node("csv", csv_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {
        "general": "general",
        "domain": "domain",
        "csv": "csv",
    },
)

workflow.add_edge("general", END)
workflow.add_edge("domain", END)
workflow.add_edge("csv", END)

app = workflow.compile()


# Streamlit

# st.set_page_config(page_title="LLM Router", page_icon="🤖")

# st.title("🚦 Multi-LLM Router with LangGraph")

# query = st.text_area("Enter your query:")

# if st.button("Run"):
#     if query.strip():
#         with st.spinner("Thinking..."):
#             result = app.invoke({"query": query})
#         st.subheader("Response:")
#         st.write(result["response"])
#     else:
#         st.warning("Please enter a query.")

import streamlit as st

# Page setup
st.set_page_config(
    page_title="Vishwakarma Art AI Assistant",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for earthy / handicraft theme
st.markdown("""
    <style>
        body {
            background-color: #f7f4ef;
        }
        .main {
            background-color: #fffaf3;
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #e0d4c2;
        }
        h1 {
            color: #6b4226;
            text-align: center;
            font-family: 'Georgia', serif;
        }
        h2, h3, h4 {
            color: #8b5e3c;
            font-family: 'Georgia', serif;
        }
        .stTextArea textarea {
            border: 2px solid #8b5e3c !important;
            border-radius: 10px !important;
            background-color: #fdfbf7 !important;
        }
        .stButton>button {
            background-color: #6b4226;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            border: none;
            padding: 0.7em 1.5em;
        }
        .stButton>button:hover {
            background-color: #4e2f1a;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🪵 Vishwakarma Arts AI Assistant")
st.markdown(
    """
    Welcome to **Vishwakarma Art**, your trusted partner in *handcrafted wooden furniture & decor*.  
    Ask me about our **craftsmanship, designs, materials, or sales insights** — I’ll help you find the right answer.
    """
)

# Query input
query = st.text_area("💬 What would you like to know?", height=120)

# Button + Response
if st.button("✨ Get Answer"):
    if query.strip():
        with st.spinner("Crafting your response... 🛠️"):
            result = app.invoke({"query": query})
        st.subheader("🖌️ Response")
        st.success(result["response"])
    else:
        st.warning("⚠️ Please enter a query to continue.")







