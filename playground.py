from bot import llm


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    print("LLM: "+llm.query(user_input)+"\n")
    