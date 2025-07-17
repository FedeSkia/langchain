from IPython.core.display import Markdown
from IPython.core.display_functions import display
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    FewShotChatMessagePromptTemplate
from langchain_ollama import OllamaLLM

prompt = """
Answer the user's query based on the context below.
If you cannot answer the question using the
provided information answer with "I don't know".

Context: {context}
"""

context = """Aurelio AI is an AI company developing tooling for AI
engineers. Their focus is on language AI with the team having strong
expertise in building AI agents and a strong background in
information retrieval.

The company is behind several open source frameworks, most notably
Semantic Router and Semantic Chunkers. They also have an AI
Platform providing engineers with tooling to help them build with
AI. Finally, the team also provides development services to other
organizations to help them bring their AI tech to market.

Aurelio AI became LangChain Experts in September 2024 after a long
track record of delivering AI solutions built with the LangChain
ecosystem."""

query = "what does Aurelio AI?"

new_system_prompt = """
Answer the user's query based on the context below.
If you cannot answer the question using the
provided information answer with "I don't know".

Always answer in markdown format. When doing so please
provide headers, short summaries, follow with bullet
points, then conclude.

Context: {context}
"""


def context_prompt():
    # the prompt has a placeholder {context}. prompt is only known by the LLM
    system_message: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template(prompt)
    human_message: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template('{query}')
    prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages([
        system_message,
        human_message
    ])
    # the input variables are query and context
    print("Prompt Input variables:", prompt_template.input_variables)
    print("Prompt Messages:", prompt_template.messages)
    messages: list[BaseMessage] = prompt_template.format_messages(query=query, context=context)
    llm = OllamaLLM(model="tinyllama:1.1b")
    response: str = llm.invoke(messages)
    print(response)


def few_shot_prompting():
    """We want to instruct the LLM on how to behave when a user asks a question.
    In this case we specify that we want a bullet point answer well formatted"""
    examples = [
        {
            "input": "Can you explain gravity?",
            "output": (
                "## Gravity\n\n"
                "Gravity is one of the fundamental forces in the universe.\n\n"
                "### Discovery\n\n"
                "* Gravity was first discovered by Sir Isaac Newton in the late 17th century.\n"
                "* It was said that Newton theorized about gravity after seeing an apple fall from a tree.\n\n"
                "### In General Relativity\n\n"
                "* Gravity is described as the curvature of spacetime.\n"
                "* The more massive an object is, the more it curves spacetime.\n"
                "* This curvature is what causes objects to fall towards each other.\n\n"
                "### Gravitons\n\n"
                "* Gravitons are hypothetical particles that mediate the force of gravity.\n"
                "* They have not yet been detected.\n\n"
                "**To conclude**, Gravity is a fascinating topic and has been studied extensively since the time of Newton.\n\n"
            )
        },
        {
            "input": "What is the capital of France?",
            "output": (
                "## France\n\n"
                "The capital of France is Paris.\n\n"
                "### Origins\n\n"
                "* The name Paris comes from the Latin word \"Parisini\" which referred to a Celtic people living in the area.\n"
                "* The Romans named the city Lutetia, which means \"the place where the river turns\".\n"
                "* The city was renamed Paris in the 3rd century BC by the Celtic-speaking Parisii tribe.\n\n"
                "**To conclude**, Paris is highly regarded as one of the most beautiful cities in the world and is one of the world's greatest cultural and economic centres.\n\n"
            )
        }
    ]

    # create a prompt with placeholders
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    # here we are going to replace the placeholders with the values in a "few shot prompt"
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    print("Few shot prompt:", few_shot_prompt)
    prompt: list[BaseMessage] = ChatPromptTemplate.from_messages([
        ("system", new_system_prompt),
        few_shot_prompt,
        ("user", "{query}"),
    ]).format_messages(query=query, context=context)

    llm = OllamaLLM(model="llama3.1:8b")
    response: str = llm.invoke(prompt)
    print(response)


if __name__ == '__main__':
    few_shot_prompting()
