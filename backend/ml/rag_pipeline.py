import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_DIR = os.path.join(RAG_DIR, "faiss_index")
KNOWLEDGE_BASE_PATH = os.path.join(RAG_DIR, "nasa_guidelines.txt")

# Ensure the mock NASA behavioral health guidelines document exists
def create_mock_nasa_guidelines():
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        content = """NASA Behavioral Health Guidelines for Deep Space Operations:
1. Cognitive Reframing: When encountering communication delays, operators should explicitly acknowledge the delay rather than repeatedly checking for responses. Reframe the waiting period as autonomous operation time.
2. Progressive Muscle Relaxation (PMR): Astronauts demonstrating high physiological stress (e.g., elevated EDA, respiratory amplitude) should engage in 5-minute PMR sessions. Focus on tensing and releasing core muscle groups.
3. Emotional Labeling: Clearly stating 'I feel overwhelmed by the current task load' reduces amygdala activation. The AI companion should encourage the crew member to articulate specific emotions rather than general frustration.
4. Sleep Hygiene in Microgravity: Maintain strict circadian alignment. If mean skin temperature and HRV indicate poor recovery, mandate a 9-hour dark period. Avoid blue light 2 hours before the scheduled sleep period.
5. Autonomy Support: Provide choices rather than directives. When an astronaut is stressed, offer 2-3 viable intervention options (e.g., 'Would you prefer to review the PMR protocol or take a 10-minute visual break?').
6. Social Connection: Despite delays, asynchronous audio/video messages to family significantly lower cortisol levels. Recommend drafting a message if the isolation index rises.
"""
        with open(KNOWLEDGE_BASE_PATH, "w") as f:
            f.write(content)

def get_vectorstore():
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(VECTORSTORE_DIR):
        print("Loading existing FAISS vectorstore...")
        return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    
    print("Creating FAISS vectorstore from mock NASA guidelines...")
    create_mock_nasa_guidelines()
    
    loader = TextLoader(KNOWLEDGE_BASE_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    
    return vectorstore

def setup_rag_chain():
    """Sets up the Retrieval-Augmented Generation chain using local Llama 3 via Ollama."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Needs Ollama installed locally and `llama3:latest` model pulled
    try:
        llm = Ollama(model="llama3:latest")
    except Exception as e:
        print("Failed to initialize Ollama. Is it running locally? Error:", e)
        # Fallback to a dumb echo if Ollama is not configured
        llm = None
        
    template = """You are MAITRI, an onboard AI health companion for deep-space astronauts. 
You are highly empathetic, clinical, and reassuring. Speak directly to the astronaut.
Provide advice based STRICTLY on the following NASA Behavioral Health Guidelines context. DO NOT hallucinate.

Guidelines Context:
{context}

{journal_context}
If journal context is present, start your response by explicitly referencing their recent logged thoughts. Example: "I remember earlier you felt..."

{chat_history}

Current Stress Level: {stress_level}
Astronaut Message: {question}

MAITRI Response:"""
    
    prompt_template = PromptTemplate(
        template=template, input_variables=["context", "journal_context", "chat_history", "question", "stress_level"]
    )
    
    return retriever, llm, prompt_template

def generate_rag_response(message: str, stress_tier: str, journal_context: str = "", chat_history: str = "") -> str:
    retriever, llm, prompt_template = setup_rag_chain()
    
    # Retrieve relevant docs
    docs = retriever.invoke(message)
    context_str = "\n".join([d.page_content for d in docs])
    
    if llm is None:
        return f"[Ollama Llama3 Fallback]: Based on local guidelines: {context_str}. Please ensure Ollama is installed."
        
    # Generate response
    formatted_prompt = prompt_template.format(
        context=context_str, 
        journal_context=journal_context,
        chat_history=chat_history,
        question=message, 
        stress_level=stress_tier
    )
    
    try:
        response = llm.invoke(formatted_prompt)
        return response
    except Exception as e:
        return f"Error communicating with local LLM. Check if Ollama is running. System retrieved context: {context_str}"
