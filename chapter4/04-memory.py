from typing import Optional, Dict, Any, List

from langchain_core.callbacks import StreamingStdOutCallbackHandler, BaseCallbackHandler
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langchain_ollama import OllamaLLM

chat_map = {}


class CallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print("=" * 50)
        print("INPUT RECEIVED FROM OLLAMA:")
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i + 1}: {prompt}")
        print("=" * 50)


def memory():
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


if __name__ == '__main__':
    memory()
