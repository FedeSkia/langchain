from langchain_community.llms.ollama import Ollama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_tool_calling_agent
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor

prompt = ChatPromptTemplate.from_messages([
    ("system", "you're a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

llm = ChatOllama(
    model="llama3.1:8b", temperature=0.0
)

memory = ConversationBufferMemory(
    memory_key="chat_history",  # must align with MessagesPlaceholder variable_name
    return_messages=True  # to return Message objects
)


@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y


@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x ** y


@tool
def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x


tools = [add, subtract, multiply, exponentiate]

agent = create_tool_calling_agent(
    llm=llm, tools=tools, prompt=prompt
)

result = agent.invoke({
    "input": "what is 10.7 multiplied by 7.68?",
    "chat_history": memory.chat_memory.messages,
    "intermediate_steps": []  # agent will append it's internal steps here
})

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
result = agent_executor.invoke({"input": "what is 10.7 multiplied by 7.68?",
                       "chat_history": memory.chat_memory.messages})
print(result)
second_result = agent_executor.invoke({
    "input": "My name is James",
    "chat_history": memory
})
print(second_result)
third_result = agent_executor.invoke({
    "input": "What is nine plus 10, minus 4 * 2, to the power of 3",
    "chat_history": memory
})
print(third_result)
fourth_result = agent_executor.invoke({
    "input": "What is my name",
    "chat_history": memory
})