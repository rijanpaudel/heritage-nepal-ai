"""
Test script to verify Heritage Nepal AI setup
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

def test_api_key():
    """Check if OpenAI API key is loaded"""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith("sk-"):
        print("✓ API key loaded successfully")
        return True
    else:
        print("✗ API key not found or invalid")
        return False

def test_openai_connection():
    """Test connection to OpenAI"""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke("Say 'Nepal Heritage AI is ready!'")
        print(f"✓ OpenAI connection successful: {response.content}")
        return True
    except Exception as e:
        print(f"✗ OpenAI connection failed: {e}")
        return False

def test_embeddings():
    """Test embedding generation"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        test_embedding = embeddings.embed_query("Traditional Newari architecture")
        print(f"✓ Embeddings working (vector dimension: {len(test_embedding)})")
        return True
    except Exception as e:
        print(f"✗ Embeddings failed: {e}")
        return False

def test_vector_store():
    """Test Chroma vector database"""
    try:
        # Create temporary vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        test_docs = ["Nepal has rich cultural heritage", "Kathmandu Valley has ancient temples"]
        vector_store = Chroma.from_texts(
            texts=test_docs,
            embedding=embeddings,
            collection_name="test_collection"
        )
        results = vector_store.similarity_search("temples in Nepal", k=1)
        print(f"✓ Vector database working (found: '{results[0].page_content}')")
        return True
    except Exception as e:
        print(f"✗ Vector store failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Heritage Nepal AI Setup...\n")
    
    tests = [
        test_api_key,
        test_openai_connection,
        test_embeddings,
        test_vector_store
    ]
    
    results = [test() for test in tests]
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    if all(results):
        print("🎉 Setup complete! Ready to build Heritage Nepal AI")
    else:
        print("⚠️  Some tests failed. Check errors above.")