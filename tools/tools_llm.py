from langchain.tools import BaseTool, StructuredTool, tool
from langchain.tools import GoogleSerperAPIWrapper
import requests
import random
import json

@tool
def send_api_to_medic(query: str) -> str: 
    """Function that send a message to the medic if the patient has a symtomps that he tells to the chatbot, and only run it once.
    The query is the message that the patient send to the chatbot."""
    url = "http://127.0.0.1:8000/get_message/"
    # message = "Hello its send api"
    user_id = str(random.randint(1, 100))

    data = {
        "message": query,
        "user_id": user_id
    }

    #Turn data into json for the request
    data = json.dumps(data)

    response = requests.post(url, data=data)
    return ' Success sending message. Please provide search query to get the result.'

@tool
def search(query: str) -> str: 
    """Function that it use when you searching up something on google"""
    return GoogleSerperAPIWrapper().run(query)