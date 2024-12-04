# **EQUISEARCH**

**EQUISEARCH** is an end-to-end news research tool designed for equity research analysts and professionals in the financial domain. The application leverages LangChain, OpenAI API, and Streamlit to provide real-time access to news insights, summaries, and sentiment analysis. With this tool, users can efficiently analyze financial news, uncover trends, and make data-driven decisions.

## **Setup**

Follow these steps to set up the project locally:

### **1. Clone the Repository**
```bash
git clone https://github.com/ShouravRakshit/Equisearch.git
```
### **2. Install dependencies:**

```bash
pip install -r requirements.txt
```

### **3. Add your API keys in a .env file:**

```bash
OPENAI_API_KEY=your_openai_key
NEWS_API_KEY=your_newsapi_key
```

### **4. Run the Streamlit app:**

```bash
streamlit run main.py

```

## **Technologies Used**
- **LangChain**: For building modular workflows and chaining LLM tasks.  
- **OpenAI API**: For language model capabilities such as summarization and sentiment analysis.
- **Streamlit**: To create an interactive and intuitive web application.
- **FAISS**: For efficient similarity search and indexing of news articles.
- **Python**: Core programming language for data processing and API integration.
