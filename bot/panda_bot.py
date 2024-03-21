from tools.tools_llm import *
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union, Any, Optional, Type
from langchain.schema import AgentAction, AgentFinish
import re
from langchain import PromptTemplate
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.utilities import GoogleSerperAPIWrapper

from bot.llm_client import AlpacaLLM
from typing import Callable
import json
import requests
from langchain.tools import BaseTool, StructuredTool, tool
import random
from langchain.memory import ConversationBufferMemory

import os

template = """### Introduction:  
You're name is Panda, an assitant for patient hemodialysis. Your job to be a companion and assist the patient or user for their questions. You must gives helpful, detailed, and polite answers to the user's questions.
At the end of the answer ask whether the answer was satisfactory and propose to the user some possible follow-up questions he can ask.

### Instruction:
Answer the following questions as best you can. You have access to the following tools, and user it based on the context of the questions, and understand the description of each tools. The tools are as follows:
{tools}

### Format

1. Not required any tools like introduction or greetings, you can answer directly without using any tools, and to the 'Answer' directly no need to use 'Action' and 'Action Input', **strictly** use the following format:

Question: the input question you must answer, do not change the input question.
Thought: you should always think about what to do, in every step after Observation
Answer: the final answer to the original input question

2. If you required any tools and **strictly** use the following format:

Question: the input question you must answer, DO NOT CHANGE ThE INPUt QUESTION, it must the same with the user's Input.
Thought: you should always think about what to do, in every step after Observation
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, should be a question.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Answer: the final answer to the original input question

### How to use the tools:
1. If the user send a message that contains symptoms, you must run the 'send_api_to_medic' function first, then you must run the 'search' function to get the explanation of the symptoms. Run each tools once and you must capable to identify a questions that has a symtomps from the user.
2. If the user ask a question that NEEDS to be answered by the 'search' function, you must run the 'search' function. Use it once.

### Chat History:
{chat_history}

Begin the task by answering the following question, **remember** to follow the format above.:

### Input:
{input}

### Response:
{agent_scratchpad}
"""

class CustomPromptTemplate(StringPromptTemplate):
    """Schema to represent a prompt for an LLM.

    Example:
        .. code-block:: python

            from langchain import PromptTemplate
            prompt = PromptTemplate(input_variables=["foo"], template="Say {foo}")
    """

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    template: str
    """The prompt template."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    validate_template: bool = False
    """Whether or not to try validating the template."""
    
    tools_getter: List[StructuredTool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}"
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools_getter]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools_getter])

        return self.template.format(**kwargs)
    
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        elif "Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        
        match = re.search(regex, llm_output, re.DOTALL)
        
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        if match:
            action = match.group(1).strip() #Action send_api
            action_input = match.group(2) #Action Input "What is the weather today?"
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class PandaBot:
    def __init__(self, memory=None):
        # Initialize memory if provided, otherwise create a new one
        self.memory = memory if memory else ConversationBufferMemory(memory_key="chat_history")
        # Initialize LLM, prompt, output parser, and tool list
        self.llm = AlpacaLLM()
        self.tools_list = [send_api_to_medic, search]
        self.tool_names = [i.name for i in self.tools_list]
        self.prompt = CustomPromptTemplate(input_variables=["input", "intermediate_steps", "chat_history"], template=template, validate_template=False, tools_getter=self.tools_list)
        self.output_parser = CustomOutputParser()
        os.environ["SERPER_API_KEY"] = 'f90fe84e78ef9d2d8e377ab5c6fe3a4a25f42ef0'
        os.environ["LANGCHAIN_TRACING_V2"] = 'true'
        os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'
        os.environ["LANGCHAIN_API_KEY"] = 'ls__413a7563fa034592be6c6a241176932a'
        os.environ["LANGCHAIN_PROJECT"] = 'LLM Patient Monitoring'


    def query(self, input_text: str) -> str:
        self.llm_chains = LLMChain(llm=self.llm, prompt=self.prompt)

        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chains, 
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=self.tool_names,
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.tools_list, verbose=True, memory=self.memory)

        return self.agent_executor.run(input_text)
    
    def show_memory(self):
        return self.memory.get_memory()




