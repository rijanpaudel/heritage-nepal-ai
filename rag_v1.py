import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (API keys)
load_dotenv()

# Verify API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file!")

print("Heritage Nepal AI - Minimal RAG System")
print("=" * 60)


#Document loading
def load_documents(data_directory="data"):
    """
    Load all text documents from the specified directory.
    
    Args:
        data_directory: Path to folder containing .txt files
    
    Returns:
        List of Document objects with content and metadata
    """
    print(f"\nSTEP 1: Loading documents from '{data_directory}/'...")
    
    try:
        # DirectoryLoader scans folder for files matching pattern
        # loader_cls=TextLoader tells it how to read each file
        loader = DirectoryLoader(
            data_directory,
            glob="**/*.txt",  # Find all .txt files, including subdirectories
            loader_cls=TextLoader,
            show_progress=True
        )
        
        documents = loader.load()
        
        print(f"Loaded {len(documents)} documents")
        
        # Show what we loaded
        for i, doc in enumerate(documents, 1):
            filename = doc.metadata.get('source', 'unknown')
            word_count = len(doc.page_content.split())
            print(f"   {i}. {filename} ({word_count} words)")
        
        return documents
    
    except Exception as e:
        print(f" Error loading documents: {e}")
        print("   Make sure your data/ folder exists with .txt files!")
        return []


#Chunking documents(text splitting)
def chunk_documents(documents, chunk_size=500, chunk_overlap=50):

    print(f"\n STEP 2: Chunking documents...")
    print(f"   Chunk size: {chunk_size} chars (~{chunk_size//4} tokens)")
    print(f"   Overlap: {chunk_overlap} chars")
    
    # RecursiveCharacterTextSplitter tries separators in order:
    # 1. "\n\n" (paragraphs) - keeps paragraphs together
    # 2. "\n" (lines) - keeps sentences together  
    # 3. " " (words) - splits on words if needed
    # 4. "" (characters) - last resort
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Use character count
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f" Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Show example chunk
    if chunks:
        print(f"\n   Example chunk from '{chunks[0].metadata['source']}':")
        print(f"   '{chunks[0].page_content[:150]}...'")
    
    return chunks

#Embedding and vector store creation
def create_vector_store(chunks):
    
    print(f"\n STEP 3: Creating embeddings & vector store...")
    print(f"   Using OpenAI model: text-embedding-3-small")
    print(f"   Embedding {len(chunks)} chunks (this takes ~1 sec per 100 chunks)...")
    
    try:
        # Initialize OpenAI embeddings
        # text-embedding-3-small creates 1536-dimensional vectors
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        # FAISS (Facebook AI Similarity Search) stores vectors in RAM
        # from_documents() automatically:
        # 1. Calls OpenAI API to embed each chunk
        # 2. Stores vectors in searchable index
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        print(f" Vector store created with {len(chunks)} embedded chunks")
        print(f"   Storage: In-memory (FAISS)")
        print(f"   Vector dimension: 1536")
        
        return vector_store
    
    except Exception as e:
        print(f" Error creating vector store: {e}")
        print("   Check your OPENAI_API_KEY and internet connection")
        return None


#Retrieval of relevant chunks
def retrieve_relevant_chunks(vector_store, question, k=3):
   
    print(f"\n STEP 4: Retrieving relevant chunks for question...")
    print(f"   Question: '{question}'")
    print(f"   Searching for top {k} most similar chunks...")
    
    try:
        # similarity_search() embeds the question and finds nearest neighbors
        # Uses cosine similarity: measures angle between vectors
        # Similar meaning = small angle = high similarity score
        relevant_chunks = vector_store.similarity_search(
            query=question,
            k=k
        )
        
        print(f" Found {len(relevant_chunks)} relevant chunks:")
        
        # Show what was retrieved
        for i, chunk in enumerate(relevant_chunks, 1):
            source = chunk.metadata.get('source', 'unknown')
            preview = chunk.page_content[:100].replace('\n', ' ')
            print(f"   {i}. From {source}: '{preview}...'")
        
        return relevant_chunks
    
    except Exception as e:
        print(f" Error during retrieval: {e}")
        return []


#Generation of answer using LLM
def generate_answer(question, relevant_chunks):
    
    print(f"\n STEP 5: Generating answer with GPT-4...")
    
    # Combine all chunk content into context string
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    
    print(f"   Context length: {len(context)} characters")
    print(f"   LLM model: gpt-4o-mini (fast & cost-effective)")
    
    # Create prompt template
    # It instructs the LLM to use ONLY our context
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable assistant for Heritage Nepal AI, 
        specializing in Nepal's culture, tourism, and heritage sites.
        
        Answer questions ONLY based on the provided context. If the context 
        doesn't contain enough information to answer, say so clearly.
        
        Be specific, accurate, and cite details from the context when possible."""),
        
        ("user", """Context from Nepal tourism documents:
        {context}
        
        Question: {question}
        
        Answer:""")
    ])
    
    # Format the prompt with our context and question
    formatted_prompt = prompt_template.format_messages(
        context=context,
        question=question
    )
    
    try:
        # Initialize GPT-4 mini (cheaper, faster than full GPT-4)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0  # 0 = deterministic, 1 = creative
        )
        
        # Generate answer
        response = llm.invoke(formatted_prompt)
        answer = response.content
        
        print(f" Answer generated ({len(answer)} characters)")
        
        return answer
    
    except Exception as e:
        print(f" Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer due to an error."


#Complete RAG pipeline
def run_rag_pipeline(question):
    
    print("\n" + "=" * 60)
    print(f"QUERY: {question}")
    print("=" * 60)
    
    # Check if we need to rebuild the vector store
    global vector_store
    
    if 'vector_store' not in globals() or vector_store is None:
        # First run - build the vector store
        docs = load_documents()
        
        if not docs:
            return " No documents found. Please add .txt files to data/ folder."
        
        chunks = chunk_documents(docs)
        vector_store = create_vector_store(chunks)
        
        if not vector_store:
            return " Failed to create vector store."
    
    # Retrieve relevant context
    relevant_chunks = retrieve_relevant_chunks(vector_store, question, k=3)
    
    if not relevant_chunks:
        return " No relevant information found."
    
    # Generate answer
    answer = generate_answer(question, relevant_chunks)
    
    return answer


#Main execution for testing
if __name__ == "__main__":
    # Initialize vector store (done once)
    vector_store = None
    
    # Test questions to verify the system works
    test_questions = [
        "What are the main heritage sites in Kathmandu?",
        "When was Mount Everest first climbed?",
        "What is Dal Bhat and why is it important in Nepal?"
    ]
    
    print("\n Running test queries...\n")
    
    # Run each test question
    for i, question in enumerate(test_questions, 1):
        answer = run_rag_pipeline(question)
        
        print("\n" + "=" * 60)
        print(f"ANSWER {i}:")
        print("=" * 60)
        print(answer)
        print("\n")
        
        if i < len(test_questions):
            print("⏸  Press Enter for next question...")
            input()
    
    print("\n" + "=" * 60)
    print(" RAG SYSTEM TEST COMPLETE!")
    print("=" * 60)
    print("\nYou can now modify the test_questions list to ask your own questions.")
    print("Or import this file and use: run_rag_pipeline('your question')")