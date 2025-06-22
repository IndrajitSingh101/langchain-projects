import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv("../env")

def load_data():
    loader=TextLoader("../data/medium.txt")
    document=loader.load()
    print("splitting...")
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts=text_splitter.split_documents(document)
    print(f"Splitted into {len(texts)} chunks")
    embeddings=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"),model="text-embedding-3-small")
    vectorstore=PineconeVectorStore(index_name=os.getenv("INDEX_NAME"),embedding=embeddings,pinecone_api_key=os.getenv("PINECONE_API_KEY"))
    vectorstore.add_documents(texts)
    print("Documents added to vectorstore")

def main():
   #load_data()
   embeddings=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"),model="text-embedding-3-small")
   llm=ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),model="gpt-4o-mini")
   query="what is airflow?"
   #chain=PromptTemplate.from_template(template=query) | llm
   #result=chain.invoke(input={})
   #print(result)
   vectorstore=PineconeVectorStore(index_name=os.getenv("INDEX_NAME"),embedding=embeddings)
   retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
   combine_docs_chain=create_stuff_documents_chain(llm,prompt=retrieval_qa_chat_prompt)
   retrieval_qa_chain = create_retrieval_chain(
       retriever=vectorstore.as_retriever(),
       combine_docs_chain=combine_docs_chain
   )
   result=retrieval_qa_chain.invoke({"input":query})
   print(result['answer'])
if __name__ == "__main__":
    main()
    