import asyncio
from typing import cast

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessageChunk, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSerializable
from langchain_ollama import ChatOllama

from fede_llm.capstone.api.ToolsUsedCallbackHandler import ToolsUsedCallbackHandler
from tools import add, final_answer, serpapi

enabled_functions = [add, final_answer, serpapi]


class CustomAgentExecutor:
    chat_history: list[BaseMessage]

    def __init__(self, prompt: ChatPromptTemplate, function_tools: [], llm: BaseChatModel,
                 toolUsedCallback: ToolsUsedCallbackHandler, max_iterations: int = 3):
        self.__functions_tools_available = {tool.name: tool.func for tool in function_tools}
        self.chat_history = []
        self.max_iterations = max_iterations
        self.toolUsedCallback = toolUsedCallback
        self.agent: RunnableSerializable = (
                {
                    "input": lambda x: x["input"],
                    "chat_history": lambda x: x["chat_history"],
                    "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
                }
                | prompt
                | llm.bind_tools(function_tools, tool_choice="any")  # we're forcing tool use again
        )

    def sync_invoke(self, user_query: str) -> dict:
        agent_scratchpad = []
        input = {
            "input": user_query,
            "chat_history": self.chat_history,
            "agent_scratchpad": agent_scratchpad
        }
        return self.agent.invoke(input=input)

    async def async_invoke(self, user_query: str):
        agent_scratchpad = []

        llm_calls = 0
        final_answer = None
        while llm_calls < 2 and not final_answer:
            llm_input = {
                "input": user_query,
                "chat_history": self.chat_history,
                "agent_scratchpad": agent_scratchpad
            }
            ai_msgs_chunks: list[AIMessageChunk] = []
            async for token in self.agent.with_config(callbacks=[self.toolUsedCallback]).astream(llm_input):
                ai_msg_chunk: AIMessageChunk = cast(AIMessageChunk, token)
                ai_msgs_chunks.append(ai_msg_chunk)
                tool_invoked = ai_msg_chunk.tool_calls
                if tool_invoked:
                    final_answer_tool = list(filter(lambda tool: tool['name'] == "final_answer", tool_invoked))
                    if final_answer_tool:
                        final_answer = final_answer_tool[0]  # exit the while
                # yield ai_msg_chunk

            ai_msg: AIMessage = await self.assemble_message(ai_msgs_chunks)
            print("AI Message: ", ai_msg)
            self.chat_history.append(ai_msg)
            self.add_scratchpad(agent_scratchpad, ai_msg.tool_calls)
            llm_calls += 1
        self.toolUsedCallback.terminate()  # if the LLM doesnt use the final_answer tool or the number of calls are too many then just end it.

    def execute_llm_tool(self, tool_used) -> any:
        """ returns the output of the same function invoked by the LLM."""
        function_name_invoked_by_llm = tool_used["name"]
        tool_function_filtered = list(filter(lambda func: func.name == function_name_invoked_by_llm, enabled_functions))
        if tool_function_filtered:
            args_used_by_llm = tool_used["args"]
            return tool_function_filtered[0](args_used_by_llm["query"])

    def add_scratchpad(self, agent_scratchpad, tool_used):
        """ if the same tool used is not already in the scratchpad then add it """
        for tool_call in tool_used:
            if not any(scratchpad['id'] == tool_call['id'] for scratchpad in agent_scratchpad):
                tool_output = self.execute_llm_tool(tool_call)
                tool_msg = ToolMessage(
                    content=f"{tool_output}",
                    tool_call_id=tool_call["id"]
                )
                print("Adding new tool to scratchpad that was not there ", tool_msg)
                agent_scratchpad.append(tool_msg)

    async def assemble_message(self, chunks: list[AIMessageChunk]) -> AIMessage:
        ai_msg = None
        for chunk in chunks:
            if not ai_msg:
                ai_msg = chunk
            else:
                ai_msg = ai_msg + chunk
        return ai_msg


prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You're a helpful assistant. When answering a user's question "
        "you should first use one of the tools provided. After using a "
        "tool the tool output will be provided back to you. When you have "
        "all the information you need, you MUST use the final_answer tool "
        "to provide a final answer to the user. Use tools to answer the "
        "user's CURRENT question, not previous questions."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
chatModel = ChatOllama(model="qwen3:0.6b", reasoning=True, num_predict=1000, temperature=0.0)

# toolUsedCallBack = ToolsUsedCallbackHandler()
# agent = CustomAgentExecutor(prompt, enabled_functions, chatModel, toolUsedCallBack)

# async def async_handler():
#     task = asyncio.create_task(agent.async_invoke("Check the weather in Lecce"))
#     async for t in toolUsedCallBack:
#         if False: print(t)
#     await task
#
#
# asyncio.run(async_handler())
