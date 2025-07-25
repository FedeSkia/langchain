from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.utils import Output
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You're a helpful assistant. When answering a user's question "
        "you should first use one of the tools provided. After using a "
        "tool the tool output will be provided in the "
        "'scratchpad' below. If you have an answer in the "
        "scratchpad you should not use any more tools and "
        "instead answer directly to the user."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")  # This is where the LLM will reason
])

llm = ChatOllama(
    model="qwen3:8b", temperature=0.0
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


def execute_same_func_of_llm(llm_output: Output, known_functions: dict) -> any:
    """ returns the output of the same function invoked by the LLM."""
    function_name_invoked_by_llm = llm_output.tool_calls[0]["name"]
    tool_function = known_functions[function_name_invoked_by_llm]
    args_used_by_llm = llm_output.tool_calls[0]["args"]
    return tool_function(**args_used_by_llm)


tools = [add, subtract, multiply, exponentiate]

name2tool: dict = {tool.name: tool.func for tool in tools}  # Holds a collection of <name, function>


def agent_with_tool_choice_any():
    agent: RunnableSerializable = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | llm.bind_tools(tools, tool_choice="any")
        # tool_choice any tells the LLM it must use a tool to provide an aswer
    )
    tool_call: Output = agent.invoke({"input": "What is 10 + 10", "chat_history": []})
    print(
        tool_call)  # tool_Calls=[{'name': 'add', 'args': {'x': 10, 'y': 10}, 'id': 'ca4097fb-8a56-4472-96c3-d085f5f657ca', 'type': 'tool_call'}]
    function_executed_by_llm_result = execute_same_func_of_llm(tool_call, name2tool)
    print(function_executed_by_llm_result)


def agent_with_tool_choice_auto():
    agent: RunnableSerializable = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | llm.bind_tools(tools, tool_choice="auto")
    )
    first_output: Output = agent.invoke({"input": "What is 10 + 10", "chat_history": []})
    print(first_output)
    tool_exec = ToolMessage(
        content=f"The {first_output.tool_calls[0]['name']} tool returned {execute_same_func_of_llm(first_output, name2tool)}",
        tool_call_id=first_output.tool_calls[0]["id"]
    )
    second_output: Output = agent.invoke({
        "input": "What is 10 + 10",
        "chat_history": [],
        "agent_scratchpad": [first_output, tool_exec]
    })
    print(second_output)


# ------------------- Now lets use the agent with tool_choice = auto --------------------- #
agent_with_tool_choice_auto()

if __name__ == '__main__':
    #agent_with_tool_choice_any()
    agent_with_tool_choice_any()
