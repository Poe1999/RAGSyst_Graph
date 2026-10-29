from typing import TypedDict, Literal

QueryType = Literal["habr", "lenta", "arxiv", "general"]
Grade = Literal["relevant", "ambiguous", "irrelevant"]


class RAGState(TypedDict, total=False):
    query: str
    query_en: str
    query_type: QueryType
    documents: list
    web_results: list
    answer: str
    sources: list
    generation_mode: str
    grade: Grade
    retry_count: int
    steps_log: list[str]