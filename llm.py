import time
import os
import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()



class LanguageModelProcessor:
    model_type = None
    model_name = None
    memory = None
    workflow = None
    app = None
    llm = None
    memory = None
     
    def __init__(self, type=None, model=None):
        llm_mapping = { 
            'groq': ChatGroq, 
            'ollama': ChatOllama,
            'openai': ChatOpenAI 
        }

        model_names = {
            'ollama': [],
            'groq': ['llama-3.1-8b-instant', 'llama-3.1-70b-versatile'],
            'openai': ['gpt-4o', 'gpt-4o-mini']
        }

        # Select the LLM type:
        if type is not None:
            model_type = type
            llm_class = llm_mapping.get(model_type)
        else:
            print("Select the LLM model type:")
            for i, model_type in enumerate(llm_mapping.keys(), start=1):
                print(f"{i}. {model_type}")
            model_type_index = int(input("Enter the number of your choice: ")) - 1
            model_type = list(llm_mapping.keys())[model_type_index]
            llm_class = llm_mapping.get(model_type)
            
        if llm_class is None:
            raise ValueError(f'Invalid model type: {model_type}')
        
        if model_type == 'ollama':
            # Fetch model names dynamically
            ollama_models = self.fetch_ollama_models()
            # Update the model_names dictionary with the fetched models
            model_names['ollama'] = ollama_models if ollama_models else ['default_model_name']
        
        if model is None:
            print(f"Select the {model_type} model:")
            for i, model_name in enumerate(model_names[model_type], start=1):
                print(f"{i}. {model_name}")
            model_name_index = int(input("Enter the number of your choice: ")) - 1
            model_name = model_names[model_type][model_name_index]
            llm_class = llm_mapping.get(model_type)

        # Debugging: Print model_type and model_name
        print(f"Model Type: {model_type}, Model Name: {model_name}")

        # Initialize the chat model
        self.llm = llm_class(model=model_name, base_url=os.getenv("OLLAMA_BASE_URL")) if model_type == 'ollama' else \
                   llm_class(temperature=0, model_name=model_name, groq_api_key=os.getenv("GROQ_API_KEY")) if model_type == 'groq' else \
                   llm_class(temperature=0, model_name=model_name, openai_api_key=os.getenv("OPENAI_API_KEY"))

        # Define the graph state to be a list of messages
        self.workflow = StateGraph(state_schema=MessagesState)

        # Define the function that calls the model
        def call_model(state: MessagesState):
            response = self.llm.invoke(state["messages"])
            return {"messages": response}

        # Define the (single) node in the graph
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", call_model)

        # Add memory
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

        # Load the system prompt from a file
        with open('system_prompt1.txt', 'r') as file:
            system_prompt = file.read().strip()

        system_message = SystemMessage(system_prompt)

        config = {"configurable": {"thread_id": "default_thread"}}
        self.app.update_state(config, {"messages": [system_message]})

        '''
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )
        '''

    def fetch_ollama_models(self):
        try:
            response = requests.get(os.getenv("OLLAMA_BASE_URL") + "/api/tags")
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()
            # Extract model names from the response
            return [model['name'] for model in data.get('models', [])]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models from Ollama API: {e}")
            return []

    def process(self, text):
        # Prepare input messages
        input_messages = [HumanMessage(text)]
        config = {"configurable": {"thread_id": "default_thread"}}

        # Invoke the application
        output = self.app.invoke({"messages": input_messages}, config)
        response_message = output["messages"][-1]

        # Print the response
        # print(f">> LLM: {response_message.pretty_print()}")
        return response_message.content
