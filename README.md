# ⚡ ZENDS Communications — AI Customer Support Copilot

> An intelligent AI-powered customer support system built for ZENDS Communications, 
> a virtual telecom company. This project automatically understands customer queries, 
> detects intent and sentiment, and generates accurate AI responses.

---

## 📌 What is this project?

Imagine you are working at a telecom company's customer support center. 
Every day, thousands of customers send messages about:
- 💳 Billing issues
- 💰 Refund requests  
- 🔧 Technical problems
- ⚠️ Complaints
- 📦 Product inquiries

Reading and replying to each message manually is slow and tiring. 
**This AI Copilot does it automatically!**

It reads the customer message → understands what they want → 
detects if they are angry or happy → and generates a perfect reply! 🤖

---

## 🎯 Business Goals

| Goal | How We Solve It |
|------|----------------|
| Reduce response time | AI generates reply instantly |
| Handle large volumes | Automated processing |
| Consistent replies | Policy-based RAG system |
| Improve satisfaction | Sentiment-aware responses |
| No real data needed | Synthetic dataset generation |

---

## 🏗️ Project Architecture

Customer Message
↓
Preprocessing (Clean Text)
↓
Intent Classification (DistilBERT)
↓
Sentiment Analysis (RoBERTa)
↓
RAG System (FAISS + Llama 3)
↓
AI Response Generation
↓
Streamlit Dashboard

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Core programming language |
| HuggingFace Transformers | DistilBERT, RoBERTa models |
| PyTorch | Deep learning framework |
| Sentence Transformers | Text embeddings for RAG |
| FAISS | Vector database |
| LangChain | RAG pipeline |
| Groq API (Llama 3.3 70B) | Response generation |
| Streamlit | Web application |
| Pandas, NumPy | Data processing |
| Matplotlib, Seaborn | EDA visualization |
| Google Colab | Model training (GPU) |

---

## 📊 Dataset Details

| Property | Value |
|----------|-------|
| Total Records | 19,995 |
| Intent Classes | 5 (Billing, Refund, Technical, Complaint, Product Inquiry) |
| Sentiment Classes | 3 (Angry, Neutral, Happy) |
| Dataset Type | Synthetic (Auto-generated) |
| Generation Method | Template + Entity Injection + Paraphrasing |
| Balance | Perfectly balanced |

---

## 🤖 Model Details

### Intent Classification Model
| Property | Value |
|----------|-------|
| Base Model | DistilBERT (distilbert-base-uncased) |
| Task | Multi-class classification (5 classes) |
| Training | Fine-tuned on ZENDS synthetic dataset |
| Epochs | 3 |
| Accuracy | 100% |
| F1 Score | 1.00 |

### Sentiment Analysis Model
| Property | Value |
|----------|-------|
| Model | cardiffnlp/twitter-roberta-base-sentiment |
| Task | 3-class sentiment (Angry/Neutral/Happy) |
| Type | Pre-trained (no fine-tuning needed) |
| Accuracy | 74.40% |
| F1 Score | 0.73 |

### RAG System
| Property | Value |
|----------|-------|
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| LLM | Llama 3.3 70B (via Groq API) |
| Chunk Size | 500 characters |
| Retrieval | Top 3 chunks |

---

## 🚀 How to Run This Project

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/ZENDS-Copilot.git
cd ZENDS-Copilot
```

### Step 2 — Create Virtual Environment
```bash
python -m venv zends_env
zends_env\Scripts\activate    # Windows
source zends_env/bin/activate  # Mac/Linux
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add Your Groq API Key
Open `streamlit_app/interface.py` and replace:
```python
GROQ_API_KEY = "your_groq_api_key_here"
```
Get free API key from 👉 https://console.groq.com

### Step 5 — Run the App
```bash
streamlit run streamlit_app/interface.py
```

                precision    recall  f1-score

 Billing          1.00      1.00      1.00
Complaint         1.00      1.00      1.00
Product Inquiry   1.00      1.00      1.00
Refund            1.00      1.00      1.00
Technical         1.00      1.00      1.00
Overall Accuracy: 100%

### Sentiment Analysis

               precision    recall  f1-score
Angry             0.60      0.91      0.72
Happy             0.95      0.95      0.95
Neutral           0.78      0.39      0.52
Overall Accuracy: 74.40%

---

## 💡 How It Works — Simple Explanation

### Step 1️⃣ — Customer sends a message
### Step 2️⃣ — Intent Detection
AI reads the message and understands:
### Step 3️⃣ — Sentiment Detection
AI detects the emotion:
### Step 4️⃣ — RAG Retrieval
AI searches ZENDS company documents:
### Step 5️⃣ — Response Generation
Llama 3.3 generates a perfect reply:
---

## 🔮 Future Improvements

| Improvement | Impact |
|-------------|--------|
| Real customer data for training | Higher accuracy |
| Fine-tune sentiment model on ZENDS data | 74% → 90%+ |
| Multi-language support (Tamil, Hindi) | Wider reach |
| Voice input support | Better UX |
| Ticket generation system | Complete automation |
| Analytics dashboard | Business insights |
| Email integration | Full automation |

---

## 👨‍💻 Project Details

| Property | Value |
|----------|-------|
| Project Title | AI Customer Support Copilot |
| Domain | Telecom Industry, Customer Service AI |
| Company | ZENDS Communications (Virtual) |
| Platform | GUVI x HCL |

---

## 📚 References

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Groq API](https://console.groq.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)

---
