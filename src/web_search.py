from tavily import TavilyClient
from langchain_core.documents import Document
from src.config import TAVILY_API_KEY

_client = TavilyClient(api_key=TAVILY_API_KEY)


def tavily_search(query: str) -> list[Document]:
    try:
        resp = _client.search(query, max_results=5)
    except Exception as e:
        print(f"[web_search] ошибка: {e}")
        return []
    return [
        Document(
            page_content=r.get("content", ""),
            metadata={"source": "web", "url": r.get("url", "")},
        )
        for r in resp.get("results", [])
    ]