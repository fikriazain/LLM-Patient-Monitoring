from bot.panda_bot import PandaBot
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

llm = PandaBot(memory=memory)