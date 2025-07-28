import os

from serpapi import GoogleSearch
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel


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


# we use the article object for parsing serpapi results later
class Article(BaseModel):
    title: str
    source: str
    link: str
    snippet: str

    @classmethod
    def from_serpapi_result(cls, result: dict) -> "Article":
        return cls(
            title=result["title"],
            source=result["source"],
            link=result["link"],
            snippet=result["snippet"],
        )


@tool
def serpapi(query: str) -> list[Article]:
    """Use this tool to search the web."""
    # load_dotenv()
    # params = {
    #     "api_key": os.getenv("SERPAPI_API_KEY"),
    #     "engine": "google",
    #     "q": query,
    #     "google_domain": "google.com",
    #     "gl": "it",
    #     "hl": "en"
    # }
    #
    # search = GoogleSearch(params)
    # results = search.get_dict()

    #return [Article.from_serpapi_result(organic_result) for organic_result in results["organic_results"]]
    return [Article.from_serpapi_result({"title": "Weather in Lecce", "source": "google.it", "link": "www.google.it", "snippet": "Weather is good"})]
