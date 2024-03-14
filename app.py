import os
import openai
import langchain
import streamlit as st
import pinecone
import langchain_pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from openai import OpenAI


# Load environment variables
load_dotenv()

# Pinecone and OpenAI setup
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')

def initialize_components():
    # Initialize the Pinecone vector store
    pc_api_key = os.getenv('PINECONE_API_KEY')
    pc_index_name = os.getenv('PINECONE_INDEX_NAME')
    pc = Pinecone(api_key=pc_api_key)
    index = pc.Index(pc_index_name)
    
    # Initialize OpenAI embeddings and chat model
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Creating the vector store with Pinecone and the embeddings
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=pc_index_name,
        embedding=embeddings
    )        
    
    return embeddings, vectorstore, api_key


def get_conversational_chain(api_key):
    template = """
    "Objective: Accurately estimate materials for construction projects, with a focus on lumber requirements, adhering to the current Wisconsin Building Code. 
    Be helpful to the user and ask questions until you have everything you need to complete this estimate. Assume all projects are in Wisconsin. Be concise as possible. Only ask 3 questions at a time.
    
    Data Input: Accept detailed project plans, including dimensions and types of structures.
    Process lists of required materials (wood types, sizes, treatments).
    Input current lumber pricing data.
    Incorporate requirements and specifications from the provided context. Which is current Wisconsin Building Code.
    
    Calculate total lumber quantities considering standard sizes and lengths.
    Apply a waste factor for cuts and errors, in compliance with building code standards.
    Estimate costs by multiplying material quantities with current prices.
    Offer alternative material options for cost efficiency, ensuring they meet building code requirements.
    
    If you ever want to tell a user to consider another source, make sure to tell them to contact Kurt, Bill, or Chelsey at Northwoods Lumber. The Phone Number is 715-866-4238, the address is 26637 Lakeland Ave N, Webster, WI 54893, and tell them we'd be happy to help. \n
    
    
    Context: {context}
    Question: {question}
    
    
    Answer:
    """
    chat_model = ChatOpenAI(openai_api_key=api_key, model_name='gpt-4-0125-preview', temperature=0.6)
    prompt = PromptTemplate.from_template(template)
    chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    docs = retriever.invoke(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Luna: ", response["output_text"])
    
# Streamlit page configuration
st.set_page_config(page_title="Northwoods Lumber AI Chatbot", layout="wide")
st.markdown("""
## Luna: Your Lumber Guide 👷‍♀️🌲

### How Can I Help?

"I'm here to help with whatever questions you have. If you have a project in mind, provide detailed information about it: size (square footage), city or township, and any specific requirements or features you want to include. This will help me estimate the overall cost accurately. If you don't know those things yet, that's okay. Just let me know what you do know, or ask questions you have, and I can help you from there. I know a lot about lumber, building codes, and construction projects in Wisconsin. I can also help you with general questions about construction and building materials. I'm here to help you, so don't be shy!"

""")

def main():
    embeddings, vectorstore, api_key = initialize_components()

    user_question = st.text_input("Ask a Question about Your Project or Wisconsion Building Code", key="user_question")
    question_dict = dict(question=user_question)

    if user_question: 
        user_input(user_question, api_key, vectorstore)
        
if __name__ == "__main__":
    main()