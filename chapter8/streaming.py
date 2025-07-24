import asyncio

from langchain.callbacks.streaming_aiter_final_only import AsyncFinalIteratorCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSerializable, Runnable
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

agent: RunnableSerializable = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
        }
        | prompt
        | llm.bind_tools(tools, tool_choice="any")  # we're forcing tool use again
)


async def stream(query: str):
    response: Runnable = agent.with_config(callbacks=[AsyncCustomCallbackHandler()])
    async for token in response.astream({
        "input": query,
        "chat_history": [],
        "agent_scratchpad": []
    }):
        print(token, flush=True)


asyncio.run(stream("What is 10+10?"))
