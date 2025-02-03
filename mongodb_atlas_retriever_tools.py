import os
import pymongo
import logging

import boto3
from langchain_aws import BedrockEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import (
    MongoDBAtlasHybridSearchRetriever,
)
from dotenv import load_dotenv

load_dotenv()


# Setup AWS and Bedrock client
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def create_embeddings(client):
    return BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=client)


# Initialize everything
bedrock_client = get_bedrock_client()
bedrock_embeddings = create_embeddings(bedrock_client)
MONGODB_URI = os.getenv("MONGODB_URI")


class MongoDBAtlasRetrieverTools:
    """
    A class that provides various tools for interacting with a MongoDB database
    to retrieve place information and perform searches.

    Tools provided by this class:
        - `mongodb_hybrid_search`: Conducts a hybrid search combining both full-text and vector search techniques.

    Author:
        Mohammad Daoud Farooqi
    """

    def mongodb_hybrid_search(query: str) -> str:
        """
        Performs a hybrid search using MongoDB Atlas, combining both full-text and vector-based searches to
        retrieve relevant documents from multiple data sources.

        Args:
            query (str): The search query string to perform the hybrid search.

        Returns:
            str: A list of document contents retrieved by the hybrid search, with each document's content
                represented as a string.

        """

        # Connect to the MongoDB database
        mongoDBClient = pymongo.MongoClient(host=MONGODB_URI)
        logging.info("Connected to MongoDB...")

        database = mongoDBClient["video_analysis"]
        collection = database["frames_data"]

        vector_store = MongoDBAtlasVectorSearch(
            text_key="document_text",
            embedding_key="document_embedding",
            index_name="vector_index",
            embedding=bedrock_embeddings,
            collection=collection,
        )

        retriever = MongoDBAtlasHybridSearchRetriever(
            vectorstore=vector_store,
            search_index_name="full_text_search_index",
            top_k=10,
        )

        documents = retriever.invoke(query)
        return [doc.page_content for doc in documents]
