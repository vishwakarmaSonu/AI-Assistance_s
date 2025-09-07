

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








