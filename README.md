# rag-chatbot-project
A Retrieval-Augmented Generation (RAG) chatbot that answers questions by retrieving relevant context from documents (adding the prompt) and generating responses using a Large Language Model (LLM). This project demonstrates how LLMs can be grounded in real data instead of hallucinating.
RAG is used because LLMs can not be up-to-date with the information, and hence this can lead to 'hallucination' when asked something thats not in the database. Retrieval uses an extrenal database, parsing which missing data can be compensated. 
#Featuers 
PDF ingestion using PyPDFDirectoryLoader
Chunking with RecursiveCharacterTextSplitter
Embedding generation via text-embedding-3-large
Vector database using ChromaDB
Top-k retrieval based on semantic similarity
Response generation using gpt-4o-mini (OpenAI)
Interactive UI with Gradio ChatInterface + streaming responses
Answers only from retrieved knowledge chunks
#Tech Stack 
Languages & Frameworks:
Python 3.x
LangChain
Gradio
AI Models:
gpt-4o-mini for generation
text-embedding-3-large for embeddings
Databases: 
ChromaDB (local persistent vector store)

