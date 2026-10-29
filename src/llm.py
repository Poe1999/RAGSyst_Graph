from langchain_gigachat.chat_models import GigaChat
from src.config import GIGACHAT_CREDENTIALS, LLM_TEMPERATURE
from src.local_embeddings import LocalEmbeddings

_llm = None
_emb = None

def get_llm() -> GigaChat:
    global _llm
    if _llm is None:
        _llm = GigaChat(
            credentials=GIGACHAT_CREDENTIALS,
            verify_ssl_certs=False,
            temperature=LLM_TEMPERATURE,
            model="GigaChat-2",
        )
    return _llm

def get_embeddings() -> LocalEmbeddings:
    global _emb
    if _emb is None:
        _emb = LocalEmbeddings()
    return _emb