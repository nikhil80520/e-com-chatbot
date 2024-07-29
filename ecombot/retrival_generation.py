from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from ecombot.ingest import ingestdata
import os
from dotenv import load_dotenv

load_dotenv()

def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={'k': 3})
    PRODUCT_BOT_TEMPLATE = """
    
    Your ecommercebot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    YOUR ANSWER:
    
    """
    
    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)
    
    llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model='gemini-pro')
    
    chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    vstore, _ = ingestdata('done')  # Unpack the tuple correctly
    chain = generation(vstore)
    print(chain.invoke("can you tell me the best bluetooth buds?"))
