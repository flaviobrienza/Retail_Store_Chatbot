from langchain_openai.chat_models import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate, SemanticSimilarityExampleSelector
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain.callbacks import LangChainTracer
from langsmith import Client


def db_query(database, vector_db, question, openai_k, langsmith_k, example_selector):
    # Tracer
    client= Client(
    api_key=langsmith_k,
    api_url='https://api.smith.langchain.com')
    callbacks = [LangChainTracer(
        project_name='retail_industry',
        client=client)]

    # LLM
    llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_k, model='gpt-3.5-turbo-1106')   

    # Example prompt (the prompt's corpus)
    example_prompt = PromptTemplate(
        input_variables=['Question', 'SQLQuery', 'SQLResult', 'Answer'], 
        template='''
    Question: {Question},
    SQLQuery: {SQLQuery},
    SQLResult: {SQLResult}, 
    Answer: {Answer} 
    '''
    ) 
    # MySQL prompt (the prompt's prefix)
    mysql_prompt = '''  
    You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".

    Use the following format:

    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here. Use a chatting format to reply to the user
    '''
    # Few shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=['input', 'table_info', 'top_k']
    ) 

    # Chain
    sql_chain = SQLDatabaseChain.from_llm(llm=llm, db=database, callbacks=callbacks, prompt=few_shot_prompt) 

    # Execution 
    return sql_chain.invoke(question)['result']


