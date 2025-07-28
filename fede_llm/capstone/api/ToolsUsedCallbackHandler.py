import asyncio
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler


class ToolsUsedCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()
        self.final_answer_seen = False

    def __aiter__(self):
        return self

    def terminate(self):
        self.queue.put_nowait("<<DONE>>")
        return self.__anext__()

    async def __anext__(self):
        token_or_done = await self.queue.get()
        if token_or_done == "<<DONE>>":
            raise StopAsyncIteration
        if token_or_done:
            return token_or_done

    async def on_llm_new_token(self, *args, **kwargs) -> None:
        chunk = kwargs.get("chunk")
        if chunk and chunk.message.tool_calls:
            final_answer = list(filter(lambda tool_call: tool_call["name"] == "final_answer", chunk.message.tool_calls))
            if final_answer:
                self.final_answer_seen = True
        self.queue.put_nowait(kwargs.get("chunk"))

    async def on_llm_end(self, *args, **kwargs) -> None:
        if self.final_answer_seen:
            self.queue.put_nowait("<<DONE>>")
        else:
            self.queue.put_nowait("<<STEP_END>>")
