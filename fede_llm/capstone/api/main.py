import asyncio
from asyncio import Task
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from fede_llm.capstone.api.ToolsUsedCallbackHandler import ToolsUsedCallbackHandler
from fede_llm.capstone.api.agent import enabled_functions, prompt, chatModel, CustomAgentExecutor

if __name__ == '__main__':
    # initilizing our application
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # Your frontend URL
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )


    async def token_generator(content: str, result_callback: ToolsUsedCallbackHandler):
        agent = CustomAgentExecutor(prompt, enabled_functions, chatModel, result_callback)

        task: Task = asyncio.create_task(agent.async_invoke(content))
        # initialize various components to stream
        # await asyncio.gather(task, consumer(result_callback=result_callback))
        async for token in result_callback:
            try:
                if token == "<<STEP_END>>":
                    # send end of step token
                    yield "</step>"
                elif tool_calls := token.message.tool_calls:
                    if tool_name := tool_calls[0]["name"]:
                        # send start of step token followed by step name tokens
                        yield f"<step><step_name>{tool_name}</step_name>"
                    if tool_args := tool_calls[0]["args"]:
                        # tool args are streamed directly, ensure it's properly encoded
                        yield tool_args
            except Exception as e:
                print(f"Error streaming token: {e}")
                continue
        await task


    @app.post("/invoke")
    async def invoke(content: str):
        callback = ToolsUsedCallbackHandler()
        return StreamingResponse(
            token_generator(content, callback),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )


    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
