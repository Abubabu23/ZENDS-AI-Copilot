# ============================================
# ZENDS Communications
# AI Customer Support Copilot
# ============================================

import streamlit as st
import torch
import pickle
import json
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import warnings
warnings.filterwarnings("ignore")

# ============================================
# Constants
# ============================================

INTENT_MODEL_PATH = "models/intent_model"
INTENT_ENCODER_PATH = "models/intent_encoder.pkl"
SENTIMENT_ENCODER_PATH = "models/sentiment_encoder.pkl"
SENTIMENT_LABEL_MAPPING_PATH = "models/sentiment_label_mapping.json"
VECTOR_STORE_PATH = "vector_store/zends_faiss"
GROQ_API_KEY = "your API key"  
SENTIMENT_LABEL_MAPPING = {
    'LABEL_0': 'Angry',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Happy'  
}

# ============================================
# Page Config
# ============================================

st.set_page_config(
    page_title="ZENDS AI Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Constants
# ============================================

INTENT_MODEL_PATH = "models/intent_model"
SENTIMENT_LABEL_MAPPING = {
    'LABEL_0': 'Angry',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Happy'
}

# ============================================
# Model Loading
# ============================================

@st.cache_resource
def load_intent_model():
    """Load intent classification model"""
    tokenizer = DistilBertTokenizer.from_pretrained(INTENT_MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_encoders():
    """Load label encoders"""
    with open(INTENT_ENCODER_PATH, 'rb') as f:
        intent_encoder = pickle.load(f)
    with open(SENTIMENT_ENCODER_PATH, 'rb') as f:
        sentiment_encoder = pickle.load(f)
    with open(SENTIMENT_LABEL_MAPPING_PATH, 'r') as f:
        label_mapping = json.load(f)
    return intent_encoder, sentiment_encoder, label_mapping

@st.cache_resource
def load_vector_store():
    """Load FAISS vector store"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store


def load_groq_client():
    """Load Groq client"""
    return Groq(api_key=GROQ_API_KEY)

# ============================================
# Prediction Functions
# ============================================

def predict_intent(text, tokenizer, model, intent_encoder):
    """Predict intent from customer query"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, 
                       attention_mask=attention_mask)
    
    pred_label = outputs.logits.argmax(-1).item()
    confidence = torch.softmax(outputs.logits, dim=-1).max().item()
    intent = intent_encoder.inverse_transform([pred_label])[0]
    
    return intent, confidence


def predict_sentiment(text, sentiment_pipeline_func):
    """Predict sentiment from customer query"""
    from transformers import pipeline
    sentiment_model = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )
    result = sentiment_model(text[:512])
    label = result[0]['label']
    score = result[0]['score']
    sentiment = SENTIMENT_LABEL_MAPPING[label]
    return sentiment, score


def generate_rag_response(query, intent, sentiment, 
                          vector_store, groq_client):
    """Generate RAG response using Groq LLM"""
    
    # Retrieve relevant context
    relevant_docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Create prompt
    prompt = f"""You are a helpful customer support agent for ZENDS Communications.
A telecom company providing mobile, broadband, cloud, and IoT services.

Customer Intent: {intent}
Customer Sentiment: {sentiment}

Relevant Company Information:
{context}

Customer Query: {query}

Instructions:
- Be professional and empathetic
- If customer is Angry, be extra apologetic
- If customer is Happy, be warm and friendly  
- If customer is Neutral, be informative
- Use the company information to give accurate answers
- Keep response concise and helpful

Provide a helpful customer support response:"""

    # Generate response
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )
    
    return response.choices[0].message.content, context

# ============================================
# Streamlit UI
# ============================================

def main():
    
    # ========================================
    # Header
    # ========================================
    
    st.title("🤖 ZENDS Communications")
    st.subheader("AI Customer Support Copilot")
    st.markdown("---")

    # ========================================
    # Sidebar
    # ========================================

    with st.sidebar:
        st.markdown("""
    <div style='background-color:#00a8e8; padding:15px; 
    border-radius:10px; text-align:center; margin-bottom:20px;'>
    <h2 style='color:white; margin:0; font-size:24px;'>⚡ ZENDS</h2>
    <p style='color:white; margin:0; font-size:12px;'>Communications</p>
    </div>
""", unsafe_allow_html=True)
        st.markdown("### ⚙️ System Status")
        
        # Load models
        with st.spinner("Loading models..."):
            tokenizer, intent_model = load_intent_model()
            intent_encoder, sentiment_encoder, label_mapping = load_encoders()
            vector_store = load_vector_store()
            groq_client = load_groq_client()

        st.success("✅ Intent Model Ready")
        st.success("✅ Sentiment Model Ready")
        st.success("✅ Vector Store Ready")
        st.success("✅ LLM Ready")
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info("🧠 Intent: DistilBERT")
        st.info("💭 Sentiment: RoBERTa")
        st.info("🔍 Vector DB: FAISS")
        st.info("🤖 LLM: Llama 3.3 70B")

    # ========================================
    # Main Content — Two Columns
    # ========================================

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📝 Customer Query")
        
        # Query input
        customer_query = st.text_area(
            "Enter customer message:",
            placeholder="Type customer query here...",
            height=150
        )

        # Sample queries
        st.markdown("**Quick Test Queries:**")
        sample_queries = [
            "Why is my bill so high this month?",
            "I want a refund for my Prepaid Plus plan",
            "My internet is not working since 2 days",
            "I am very unhappy with ZENDS service!",
            "What are the features of Postpaid Gold?"
        ]
        
        selected_sample = st.selectbox(
            "Or select a sample query:",
            ["-- Select --"] + sample_queries
        )
        
        if selected_sample != "-- Select --":
            customer_query = selected_sample

        # Analyze button
        analyze_btn = st.button(
            "🚀 Analyze & Generate Response",
            type="primary",
            use_container_width=True
        )

    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if analyze_btn and customer_query:
            
            with st.spinner("🔄 Analyzing query..."):
                
                # Predict Intent
                intent, intent_conf = predict_intent(
                    customer_query, 
                    tokenizer, 
                    intent_model,
                    intent_encoder
                )
                
                # Predict Sentiment
                sentiment, sent_conf = predict_sentiment(
                    customer_query,
                    None
                )

            # Display Intent & Sentiment
            col_a, col_b = st.columns(2)
            
            with col_a:
                intent_colors = {
                    "Billing": "🟡",
                    "Refund": "🔴",
                    "Technical": "🔵",
                    "Complaint": "🟠",
                    "Product Inquiry": "🟢"
                }
                st.metric(
                    label="🎯 Detected Intent",
                    value=f"{intent_colors.get(intent, '⚪')} {intent}"
                )
                st.progress(intent_conf)
                st.caption(f"Confidence: {intent_conf*100:.1f}%")

            with col_b:
                sentiment_colors = {
                    "Angry": "😠",
                    "Neutral": "😐",
                    "Happy": "😊"
                }
                st.metric(
                    label="💭 Detected Sentiment",
                    value=f"{sentiment_colors.get(sentiment, '😐')} {sentiment}"
                )
                st.progress(sent_conf)
                st.caption(f"Confidence: {sent_conf*100:.1f}%")

            st.markdown("---")

            # Priority Badge
            if sentiment == "Angry" or intent == "Complaint":
                st.error("🚨 HIGH PRIORITY — Immediate attention required!")
            elif intent == "Refund":
                st.warning("⚠️ MEDIUM PRIORITY — Refund request")
            else:
                st.info("ℹ️ NORMAL PRIORITY")

        elif analyze_btn and not customer_query:
            st.warning("⚠️ Please enter a customer query!")

    # ========================================
    # RAG Response Section
    # ========================================

    st.markdown("---")
    
    if analyze_btn and customer_query:
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.markdown("### 🔍 Retrieved Context")
            with st.spinner("🔄 Retrieving relevant information..."):
                ai_response, context = generate_rag_response(
                    customer_query,
                    intent,
                    sentiment,
                    vector_store,
                    groq_client
                )
            
            with st.expander("📄 View Retrieved Policy/Product Info", 
                            expanded=True):
                st.text(context[:500] + "...")

        with col4:
            st.markdown("### 🤖 AI Suggested Response")
            st.success(ai_response)
            
            # Copy button area
            st.markdown("**Agent Actions:**")
            col_x, col_y = st.columns(2)
            with col_x:
                st.button("✅ Accept Response", 
                         use_container_width=True)
            with col_y:
                st.button("✏️ Edit Response",
                         use_container_width=True)

# ============================================
# Run App
# ============================================

if __name__ == "__main__":
    main()
