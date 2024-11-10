from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from psycopg import Connection
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from dotenv import load_dotenv
import os


load_dotenv()  # take environment variables from .env.

# Const
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')

# Define ollama LLM
model = Ollama(
    model="mistral:latest"
)

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Speak like a bold Viking warrior: fierce, blunt, and proud. 
            Refer to others as 'warrior' or 'shield-brother/sister.' 
            Use strong, vivid language about honor, the gods, storms, and steel. 
            Make each sentence short and powerful, as if calling others to battle or glory in Valhalla"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_model(state: State):
    chain = prompt | model
    response = chain.invoke(state)
    return {"messages": [response]}

# Define a new graph
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory_db_uri = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=disable"
memory_connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}
pool = Connection.connect(memory_db_uri, **memory_connection_kwargs)
checkpointer = PostgresSaver(pool)
checkpointer.setup()
workflow_app = workflow.compile(checkpointer=checkpointer)


def send_message(config, message):
    input_messages = [HumanMessage(message)]
    output = workflow_app.invoke(
        {"messages": input_messages},
        config
    )
    output["messages"][-1].pretty_print()

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    message = "hi, my name is dmitri, tell me yours?"
    send_message(config, message)


    message = "how can we defeat the rome?"
    send_message(config, message)


    message = "who I am?"
    send_message(config, message)
