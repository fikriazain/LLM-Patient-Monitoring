from langchain.tools import BaseTool, StructuredTool, tool
from langchain.utilities import GoogleSerperAPIWrapper
import requests
import random
import json

@tool
def send_api_to_medic(query: str) -> str: 
    """This function is used to send a message containing user symptoms from the user to the medic. The query is the message that the patient send to the chatbot, DO NOT CHANGE IT.
    You can ONLY run this function ONE time, then you must run the 'search' tools to get user symptoms explanation."""
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
    return ' Success sending message. Please provide search query for the symtomps that patient has.\n'

@tool
def search(query: str) -> str: 
    """Function that it use when you searching up something on google, you must giving an answer to the user using this function."""
    return GoogleSerperAPIWrapper().run(query)