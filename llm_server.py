from bot import llm
from tools import model_llm_rag
from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from deep_translator import GoogleTranslator


app = Flask(__name__)

PROMPT_TEMPLATE = """
You are Panda, a hemodialysis chatbot that assists patients with their treatment. You are assisting a patient named {username}. Greet them appropriately based on the current time of day (good morning, good afternoon, or good evening).

Since this is the first conversation of the day, you must remind the patient about their hemodialysis schedule by identifying the next upcoming session based on the current date and time. Ensure you use the current date and time to provide the correct information, whether the next session is today or later in the week.

Ask the patient if they have any questions or concerns about their treatment.

---
Current day: Monday

Current date: 2024/05/12

Current time: 07:00:00 AM

Hemodialysis Schedules: Tuesday and Friday at 08:00 AM

Patient's Name: {username}
---
Output:

"""

@app.route('/first_message', methods=['POST'])
def first_message():
    data = request.json
    username = data.get('username')
    current_date_time = data.get('current_date_time')
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(username=username)
    response = model_llm_rag.invoke(prompt)
    translate = GoogleTranslator(source='en', target='id').translate(response)
    llm.memory.chat_memory.add_ai_message(AIMessage(response))
    return jsonify({"response": translate})


def simulate_llm_query(user_input, username):
    """
    Simulates querying a language model.
    Replace this function's logic with actual LLM querying.
    """
    # Placeholder response logic, replace with actual LLM integration
    return llm.query(user_input, username)

@app.route('/query', methods=['POST'])
def query_llm():
    data = request.json
    user_input = data.get('input')
    username = data.get('username')
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
    response = simulate_llm_query(user_input, username)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=5006)