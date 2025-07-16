from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import time

def setup_pinecone_index():

    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "sing2search-index")

    if not api_key:
        print("Error with Pinecone API key in .env")
        return False

    try:
        pc = Pinecone(api_key=api_key)

        index_list = [index.name for index in pc.list_indexes()]
        if index_name in index_list:
            print(f"Index {index_name} exists")

            index_info = pc.describe_index(index_name)
            print(f"DIMENSION: {index_info.dimension}")
            print(f"METRIC: {index_info.metric}")
            print(f"SPEC: {index_info.spec}")

            return True
        else:
            print(f"Index {index_name} not found")
            return False
    except Exception as e:
        print(f"Error connecting with Pinecone {str(e)}")
        return False

def test_pinecone_connection():

    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "sing2search-index")

    try:
        pc = Pinecone(api_key=api_key)

        index_info = pc.describe_index(index_name)

        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        print(f"Pinecone connected success")
        print(f"Index info: Status: {index_info.status['ready']}, Total Vectors: {stats.total_vector_count}, Dimension: {stats.dimension}")

        return True
    
    except Exception as e:
        print(f"Error connecting Pinecone: {str(e)}")
        return False
    
if __name__ == "__main__":
    print("Setting up Pinecone for Sing2Search")
    print()

    if setup_pinecone_index():
        print()
        print("Testing connection")
        test_pinecone_connection()
        print()
        print("Pinecone setup completed")
    else:
        print()
        print("Pinecone setup failed")



