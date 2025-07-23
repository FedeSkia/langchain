from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.utils import Output
import json

from langchain_core.tools import tool
from langchain_ollama import ChatOllama


class CustomAgentExecutor:
    chat_history: list[BaseMessage]

    def __init__(self, prompt: ChatPromptTemplate, function_tools: [], llm: BaseChatModel, max_iterations: int = 3):
        self.__functions_tools_available = {tool.name: tool.func for tool in function_tools}
        self.chat_history = []
        self.max_iterations = max_iterations
        self.agent: RunnableSerializable = (
                {
                    "input": lambda x: x["input"],
                    "chat_history": lambda x: x["chat_history"],
                    "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
                }
                | prompt
                | llm.bind_tools(function_tools, tool_choice="any")  # we're forcing tool use again
        )

    def invoke(self, user_query: str) -> dict:
        # invoke the agent but we do this iteratively in a loop until
        # reaching a final answer
        count = 0
        agent_scratchpad = []
        while count < self.max_iterations:
            # invoke a step for the agent to generate a tool call
            output: Output = self.agent.invoke({
                "input": user_query,
                "chat_history": self.chat_history,
                "agent_scratchpad": agent_scratchpad
            })
            # add initial tool call to scratchpad
            agent_scratchpad.append(output)
            # otherwise we execute the tool and add it's output to the agent scratchpad
            tool_name = output.tool_calls[0]["name"]
            tool_args = output.tool_calls[0]["args"]
            tool_call_id = output.tool_calls[0]["id"]
            function_executed_by_llm_output = self.execute_same_func_of_llm(output)
            # add the tool output to the agent scratchpad
            tool_exec = ToolMessage(
                content=f"{function_executed_by_llm_output}",
                tool_call_id=tool_call_id
            )
            agent_scratchpad.append(tool_exec)
            # add a print so we can see intermediate steps
            print(f"{count}: {tool_name}({tool_args}). LLM Response: ", output)
            count += 1
            # if the tool call is the final answer tool, we stop
            if tool_name == "final_answer":
                break
        # add the final output to the chat history
        final_answer = ["answer"]
        self.chat_history.extend([
            HumanMessage(content=user_query),
            AIMessage(content=final_answer)
        ])
        # return the final answer in dict form
        return json.dumps(function_executed_by_llm_output)

    def execute_same_func_of_llm(self, llm_output: Output) -> any:
        """ returns the output of the same function invoked by the LLM."""
        function_name_invoked_by_llm = llm_output.tool_calls[0]["name"]
        tool_function = self.__functions_tools_available[function_name_invoked_by_llm]
        args_used_by_llm = llm_output.tool_calls[0]["args"]
        return tool_function(**args_used_by_llm)


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


@tool
def final_answer(answer: str, tools_used: list[str]) -> str:
    """Use this tool to provide a final answer to the user.
    The answer should be in natural language as this will be provided
    to the user directly. The tools_used must include a list of tool
    names that were used within the `scratchpad`. This tool must be included if its used in the list of tools used
    """
    return {"answer": answer, "tools_used": tools_used}


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

tools = [final_answer, add, subtract, multiply, exponentiate]
agent_executor = CustomAgentExecutor(prompt, tools, ChatOllama(model="qwen3:8b", temperature=0.0))
print(agent_executor.invoke(user_query="What is 10 + 10"))
