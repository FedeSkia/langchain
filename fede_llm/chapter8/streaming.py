import asyncio

from langchain.callbacks.streaming_aiter_final_only import AsyncFinalIteratorCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSerializable
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:8b", temperature=0.0, streaming=True)


class QueueCallbackHandler(AsyncFinalIteratorCallbackHandler):
    """Callback handler that puts tokens into a queue."""

    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self.queue = queue
        self.final_answer_seen = False

    def copy(self):
        # Crea una nuova istanza che usa la stessa coda
        return QueueCallbackHandler(self.queue)

    async def __aiter__(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue
            token_or_done = await self.queue.get()

            if token_or_done == "<<DONE>>":
                # this means we're done
                return
            if token_or_done:
                yield token_or_done

    async def on_llm_new_token(self, *args, **kwargs) -> None:
        """Put new token in the queue."""
        # print(f"on_llm_new_token: {args}, {kwargs}")
        chunk = kwargs.get("chunk")
        if chunk:
            # check for final_answer tool call
            if tool_calls := chunk.message.additional_kwargs.get("tool_calls"):
                if tool_calls[0]["function"]["name"] == "final_answer":
                    # this will allow the stream to end on the next `on_llm_end` call
                    self.final_answer_seen = True
        self.queue.put_nowait(kwargs.get("chunk"))
        return

    async def on_llm_end(self, *args, **kwargs) -> None:
        """Put None in the queue to signal completion."""
        # print(f"on_llm_end: {args}, {kwargs}")
        # this should only be used at the end of our agent execution, however LangChain
        # will call this at the end of every tool call, not just the final tool call
        # so we must only send the "done" signal if we have already seen the final_answer
        # tool call
        if self.final_answer_seen:
            self.queue.put_nowait("<<DONE>>")
        else:
            self.queue.put_nowait("<<STEP_END>>")
        return


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

    async def invoke(self, user_query: str, streamer: QueueCallbackHandler) -> dict:
        # invoke the agent but we do this iteratively in a loop until
        # reaching a final answer
        count = 0
        agent_scratchpad = []
        while count < self.max_iterations:
            tool_call = await self.stream(query=user_query, streamer_callback=streamer,
                                          agent_scratchpad=agent_scratchpad)
            agent_scratchpad.append(tool_call)
            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_call_id = tool_call.tool_call_id
            tool_out = {tool.name: tool.func for tool in tools}[tool_name](**tool_args)
            tool_exec = ToolMessage(
                content=f"{tool_out}",
                tool_call_id=tool_call_id
            )
            count += 1
            agent_scratchpad.append(tool_exec)
            if tool_name == "final_answer":
                break

        final_answer = tool_out["answer"]
        self.chat_history.extend([
            HumanMessage(content=user_query),
            AIMessage(content=final_answer)
        ])
        # return the final answer in dict form
        return tool_args

    async def stream(self, query: str, streamer_callback: QueueCallbackHandler, agent_scratchpad: []) -> AIMessage:
        response = self.agent.with_config(callbacks=[streamer_callback])

        output = None
        async for token in response.astream({
            "input": query,
            "chat_history": self.chat_history,
            "agent_scratchpad": agent_scratchpad
        }):
            if output is None:
                output = token
            else:
                output += token
        return AIMessage(
            content=output.content,
            tool_calls=output.tool_calls,
            tool_call_id=output.tool_calls[0]["id"]
        )


queue = asyncio.Queue()
streamer: QueueCallbackHandler = QueueCallbackHandler(queue)
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


tools = [final_answer, add, subtract, multiply, exponentiate]

agent_executor = CustomAgentExecutor(prompt=prompt, function_tools=tools, llm=llm)


async def consume():
    async for token in streamer:
        print(token.text, flush=True)


async def main():
    task = asyncio.create_task(agent_executor.invoke(user_query="What is 10+10?", streamer=streamer))
    await asyncio.gather(task, consume())


asyncio.run(main())
