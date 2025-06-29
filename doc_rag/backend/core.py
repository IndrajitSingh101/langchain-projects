from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever


INDEX_NAME="langchain-doc-index"

load_dotenv("../../env")
def run_llm(query:str,chat_history:list):
    embeddings=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"),model="text-embedding-3-small")
    llm=ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),model="gpt-4o-mini")
    vectorstore=PineconeVectorStore(index_name=INDEX_NAME,embedding=embeddings)
    retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain=create_stuff_documents_chain(llm,prompt=retrieval_qa_chat_prompt)
    rephrase_prompt=hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever=create_history_aware_retriever(llm,retriever=vectorstore.as_retriever(),prompt=rephrase_prompt)
    retrieval_qa_chain=create_retrieval_chain(retriever=history_aware_retriever,
                                              combine_docs_chain=stuff_documents_chain)
    result=retrieval_qa_chain.invoke({"input":query,"chat_history":chat_history})
    new_result={
        "query":result["input"],
        "result":result["answer"],
        "source_documents":result["context"],
    }
    return new_result

def create_sources_string(sources):
    """Format a set of sources as a string with each source on a new line prefixed by a bullet point."""
    if not sources:
        return "No sources found."
    return "Sources:\n" + "\n".join(f"- {src}" for src in sorted(sources))
