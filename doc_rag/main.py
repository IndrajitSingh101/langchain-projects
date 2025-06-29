from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import streamlit as st
from backend.core import create_sources_string, run_llm
from langchain_community.document_loaders.firecrawl import FireCrawlLoader

load_dotenv("../env")



def ingest_docs():
    # Process files one by one or in very small batches
    docs_dir = "../data/langchain-docs"
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20,length_function=len)
    all_chunks = []
    # Process each HTML file individually
    for filename in os.listdir(docs_dir):
        if filename.endswith('.html'):
            file_path = os.path.join(docs_dir, filename)
            
            # Load the HTML file
            loader = UnstructuredHTMLLoader(file_path)
            documents = loader.load()
            
            # Split into small chunks
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            
            print(f"Processed {filename}: {len(chunks)} chunks")
    # Add to Pinecone in very small batches
    batch_size = 3  # Very small batch size
    # Add to Pinecone in very small batches
    embeddings=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"),model="text-embedding-3-small")
    vectorstore=PineconeVectorStore(index_name="langchain-doc-index",embedding=embeddings)
   

    # Add remaining chunks in small batches
    for i in range(batch_size, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Added batch {i//batch_size + 1} of {(len(all_chunks)-batch_size)//batch_size + 1}")

    print(f"Total chunks processed: {len(all_chunks)}")
if ("user_prompt_history" not in st.session_state
    and "chat_answers_history" not in st.session_state
    and "chat_history" not in st.session_state):
    st.session_state["user_prompt_history"]=[]
    st.session_state["chat_answers_history"]=[]
    st.session_state["chat_history"]=[]

def ingest_docs_firecrawl():
    crawl_params = {
    'crawlerOptions': {
        'limit': 1000,
    }
}
    loader=FireCrawlLoader(api_key=os.getenv('FIRE_CRAWL_API_KEY'),
                           url="https://python.langchain.com/api_reference",
                           mode="crawl",
                           params=crawl_params
                           )
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20,length_function=len)
    all_chunks = []
    
    # Split into small chunks
    chunks = text_splitter.split_documents(docs)
    all_chunks.extend(chunks)
    print(f"Processed{len(chunks)} chunks")
    # Add to Pinecone in very small batches
    batch_size = 3  # Very small batch size
    # Add to Pinecone in very small batches
    embeddings=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"),model="text-embedding-3-small")
    vectorstore=PineconeVectorStore(index_name="langchain-doc-index",embedding=embeddings)
    # Add remaining chunks in small batches
    for i in range(batch_size, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Added batch {i//batch_size + 1} of {(len(all_chunks)-batch_size)//batch_size + 1}")

    print(f"Total chunks processed: {len(all_chunks)}")

def run_rag():
    st.header("Langchain Doc RAG")
    prompt=st.chat_input("Enter your prompt here..")
    
    if prompt:
        with st.spinner("Generating response..."):
            generated_response=run_llm(query=prompt,chat_history=st.session_state["chat_history"])
            sources = set([doc.metadata['source'] for doc in generated_response['source_documents']])
            formatted_responses=( f"{generated_response['result']} \n\n {create_sources_string(sources)}" )
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_responses)
            st.session_state["chat_history"].append(("human",prompt))
            st.session_state["chat_history"].append(("ai",generated_response["result"]))

    if st.session_state["chat_answers_history"]:
         for generated_response,user_query in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
              st.chat_message("user").write(user_query)
              st.chat_message("assistant").write(generated_response)
if __name__ == "__main__":
    ingest_docs_firecrawl()
    #run_rag()
    #ingest_docs()
    #result=run_llm(query="what is the difference between lanchain and lanchchain-community")
    #print(result['result'])
    #print(result['source_documents'])
