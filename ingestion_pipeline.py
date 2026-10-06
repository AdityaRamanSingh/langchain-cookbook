import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_document(docs_path = "docs"):
    """Load all the text files from docs directory"""
    print(f"Loading documents from {docs_path}....")

    # checking if docs path exist
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path } doesn't exist. Plaease create and add your company files")
    
    print("Docs found")


    # load all .txt files from the docs directory
    loader = DirectoryLoader(
            path=docs_path,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt file found in {docs_path }. Please add your company document")
    
    print(".txt files found")

    for i, doc in enumerate(documents[:2]): #show first 2 only
        print(f"\n Document {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content:{len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}") 


    return documents


# main funciton
def main():
    print("Main funciton")
# 1 Loading the files
    documents = load_document(docs_path = "docs")
    print(f"Total documents loaded: {len(documents)}")

# 2 chunking the files
# embedding & storing in vectorDB
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")


if __name__ == "__main__":
    main()