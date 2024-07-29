from langchain_astradb import AstraDBVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import pandas as pd
from ecombot.data_converter import dataconverter

load_dotenv()

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

ASTRA_DB_API_ENDPOINT=os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN=os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_KEYSPACE=os.getenv('ASTRA_DB_KEYSPACE')

embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
def ingestdata(status):
    vstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name='e_com_chatbot',
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE
    )

    if status is None:
        docs = dataconverter()  # Ensure dataconverter returns a valid iterable
        if docs is None:
            raise ValueError("dataconverter returned None. Ensure it returns a valid list of documents.")
        if not docs:
            print("Warning: dataconverter returned an empty list of documents.")
        inserted_ids = vstore.add_documents(docs)
        return vstore, inserted_ids
    else:
        return vstore, []

if __name__ == "__main__":
    vstore, inserted_ids = ingestdata(None)
    results = vstore.similarity_search('can you tell me the low budget sound basshead')
    for res in results:
        print(f"*{res.page_content} [{res.metadata}]")
