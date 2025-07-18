from typing import Optional, Dict, Any, List

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.callbacks import StreamingStdOutCallbackHandler, BaseCallbackHandler
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig, ConfigurableFieldSpec
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field

chat_map = {}


class CallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print("=" * 50)
        print("INPUT SENT TO OLLAMA:")
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i + 1}: {prompt}")
        print("=" * 50)


def memory_runnable_with_message_history():
    """ https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html """
    llm = OllamaLLM(model="llama3.1:8b", callbacks=[StreamingStdOutCallbackHandler(), CallbackHandler()])

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a helpful assistant called Zeta."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    chain = prompt_template.pipe(llm)

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        """ this method returns the InMemoryChatHistory given a session_id """
        print("Invoked with session_id:", session_id)
        if session_id not in chat_map:
            chat_map[session_id] = InMemoryChatMessageHistory()
        return chat_map[session_id]

    pipeline: RunnableWithMessageHistory = RunnableWithMessageHistory(chain, get_session_history=get_session_history,
                                                                      input_messages_key="question",
                                                                      history_messages_key="history")
    config: RunnableConfig = {
        "configurable": {
            "session_id": "123"
        }
    }
    print("request: ", pipeline)
    response = pipeline.invoke(input={"question": "Hello, my name is Fede"}, config=config)
    print("response: ", response)
    print("---------------------")
    # check if AI remembers my name
    print("request: ", pipeline)
    response = pipeline.invoke(input={"question": "What's my name?"}, config=config)
    print("response: ", response)


class BufferWindowMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    k: int = Field(default_factory=int)

    def __init__(self, k):
        super().__init__(k=k)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """  Add message to the history, removing ny messages beyond the last k messages """
        self.messages.extend(messages)
        self.messages = self.messages[-self.k:]

    def clear(self) -> None:
        self.messages = []


def conversation_buffer_window_memory():
    """ In this way we dont send the entire chat history. """

    def get_chat_history(session_id: str, k: int = 4) -> BufferWindowMessageHistory:
        if session_id not in chat_map:
            chat_map[session_id] = BufferWindowMessageHistory(k=k)
        return chat_map[session_id]

    llm = OllamaLLM(model="llama3.1:8b", callbacks=[StreamingStdOutCallbackHandler(), CallbackHandler()],
                    base_url="http://localhost:8080")

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a helpful assistant called Zeta."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    pipeline = prompt_template | llm
    pipe_with_history = RunnableWithMessageHistory(pipeline, get_session_history=get_chat_history,
                                                   input_messages_key="query",
                                                   history_messages_key="history",
                                                   history_factory_config=[
                                                       ConfigurableFieldSpec(id="session_id",
                                                                             annotation=str,
                                                                             name="Session ID",
                                                                             description="The session ID to use for "
                                                                                         "the chat history",
                                                                             default="id_default"),
                                                       ConfigurableFieldSpec(id="k",
                                                                             annotation=int,
                                                                             name="k",
                                                                             description="The number of messages to "
                                                                                         "keep in history",
                                                                             default=4)
                                                   ])

    # Here im going to mock the chat history
    session_id = "fede"
    chat_history: BufferWindowMessageHistory = BufferWindowMessageHistory(1)
    messages: list[BaseMessage] = [
        HumanMessage("Hello, my name is Federico"),
        AIMessage("Great"),
        HumanMessage("Another message"),
        AIMessage("You are stupid"),
        HumanMessage("Pfff"),
        AIMessage("Hello")
    ]
    chat_history.add_messages(messages)
    chat_map[session_id] = chat_history

    config: RunnableConfig = {
        "configurable": {
            "session_id": session_id,
            "k": 1
        }
    }
    response = pipe_with_history.invoke(input={"question": "What was the last message you sent me?"}, config=config)
    print(response)


class ConversationSummaryMessageHistory(BaseChatMessageHistory, BaseModel):
    summary: str = Field(default_factory=str)
    llm: OllamaLLM = Field(default_factory=OllamaLLM)
    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, new_messages: list[BaseMessage]) -> None:
        """Add messages to the history, removing any messages beyond
        the last `k` messages.
        """
        # construct the summary chat messages
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Given the existing conversation summary and the new messages, "
                "generate a new summary of the conversation. Ensuring to maintain "
                "as much relevant information as possible."
            ),
            HumanMessagePromptTemplate.from_template(
                "Existing conversation summary:\n{existing_summary}\n\n"
                "New messages:\n{new_messages}"
            )
        ])

        # format the messages and invoke the LLM
        new_summary = self.llm.invoke(
            summary_prompt.format_messages(
                existing_summary=self.summary,
                new_messages=[x.content for x in new_messages]
            )
        )
        # replace the existing history with a single system summary message
        self.summary = new_summary
        self.messages.extend(new_messages)

    def clear(self) -> None:
        """Clear the history."""
        self.summary = ""
        self.messages = []


def conversation_summary_memory():
    """ Next up we have `ConversationSummaryMemory`,
     this memory type keeps track of a summary of the conversation rather than the entire conversation.
     This is useful for long conversations where we don't need to keep track of the entire conversation,
     but we do want to keep some thread of the full conversation.
     """

    def get_chat_history(session_id: str, llm: OllamaLLM) -> ConversationSummaryMessageHistory:
        if session_id not in chat_map:
            chat_map[session_id] = ConversationSummaryMessageHistory(llm=llm)
        return chat_map[session_id]

    llm = OllamaLLM(model="llama3.1:8b",
                    base_url="http://localhost:8080")

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a helpful assistant called Zeta."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    pipeline = prompt_template | llm

    pipeline_with_history = RunnableWithMessageHistory(
        pipeline,
        get_session_history=get_chat_history,
        input_messages_key="query",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="session_id",
                annotation=str,
                name="Session ID",
                description="The session ID to use for the chat history",
                default="id_default",
            ),
            ConfigurableFieldSpec(
                id="llm",
                annotation=OllamaLLM,
                name="LLM",
                description="The LLM to use for the conversation summary",
                default=llm,
            )
        ]
    )
    config: RunnableConfig = {"configurable": {"session_id": "id_123", "llm": llm}}
    response = pipeline_with_history.invoke(
        input={"question": "Hi, my name is Fede"},
        llm=llm,
        config=config
    )
    history = chat_map["id_123"]
    history.add_messages([HumanMessage(content="Hi, my name is Fede"), AIMessage(content=response)])
    print("Summary generated:", chat_map["id_123"].summary)

    for msg in [
        "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory.",
        "Buffer memory just stores the entire conversation",
        "Buffer window memory stores the last k messages, dropping the rest."
    ]:
        pipeline_with_history.invoke(
            input={"question": msg},
            config=config
        )
        history = chat_map["id_123"]
        history.add_messages([HumanMessage(content=msg), AIMessage(content=response)])
        print("Summary generated: ", chat_map["id_123"].summary)

    print("Summary generated: ", chat_map["id_123"].summary)


if __name__ == '__main__':
    # memory_runnable_with_message_history()
    # conversation_buffer_window_memory()
    conversation_summary_memory()
