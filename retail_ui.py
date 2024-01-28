import streamlit as st 
from langchain_community.utilities import SQLDatabase
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from retail_helper import db_query
from langchain.prompts import SemanticSimilarityExampleSelector

from secret_key import openai_key, langsmith_key 

embedder = OpenAIEmbeddings(openai_api_key=openai_key)
vector_db = Chroma(persist_directory='./retail_vector_db', embedding_function=embedder) 
database = SQLDatabase.from_uri('mysql+pymysql://root:root@localhost:3306/atliq_tshirts')
example_selector = SemanticSimilarityExampleSelector(
vectorstore=vector_db,
k=2
)

st.header('Retail Store Assistant'+':shopping_bags:') 
query = st.text_input('Your question here')
button = st.button('Submit') 

if query and button:
    st.write(db_query(database=database, 
                      vector_db=vector_db, 
                      question=query, 
                      openai_k=openai_key, 
                      langsmith_k=langsmith_key,
                      example_selector=example_selector))