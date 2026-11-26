"""Recuperação: pergunta → documentos mais relevantes do ChromaDB."""
import re
import unicodedata
from functools import lru_cache

from rank_bm25 import BM25Okapi

from config import TOP_K, MAX_DIST
from ingest import embed, get_collection

# Stopwords manuais pra nao precisar baixar NLTK dado a pouca quantidade de dados.
# Português (idioma das perguntas) + inglês (idioma dos dados de exemplo, Northwind).
# ADAPTE: troque pelo(s) idioma(s) da sua base.
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "de", "da", "do", "das", "dos", "e", "em", "no", "na",
    "nos", "nas", "para", "por", "com", "que", "qual", "quais", "quem", "como", "se", "ao",
    "the", "of", "and", "an", "in", "on", "for", "to", "with", "is", "are", "by", "from", "at",
}


def open_collection():
    """Abre a coleção e falha com mensagem clara se o índice ainda não foi criado."""
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("Índice vazio: rode `python ingest.py` antes.")
    return collection


#Busca Semantica - Dense

def dense_search(query: str, k: int = TOP_K, max_distance: float = MAX_DIST, where: dict | None = None) -> list[dict]:
    """Busca semântica: os k documentos mais próximos da pergunta (menor distância = mais parecido)."""
    query_vector = embed([query], "RETRIEVAL_QUERY")[0]
    res = open_collection().query(query_embeddings=[query_vector], n_results=k, where=where)
    return [
        {"id": id_, "texto": doc, "metadata": meta, "distancia": dist}
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
        if dist <= max_distance #Threshold do Top-K
    ]


def tokenize(text: str) -> list[str]:
    """Minúsculas, sem acentos, só palavras/números, sem stopwords: 'Côte de Blaye' → ['cote', 'blaye']."""
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return [token for token in re.findall(r"\w+", text) if token not in STOPWORDS]


@lru_cache(maxsize=1)
def bm25_index():
    """Monta o índice BM25 uma única vez, sobre os mesmos documentos e IDs gravados no Chroma."""
    data = open_collection().get(include=["documents", "metadatas"])
    index = BM25Okapi([tokenize(doc) for doc in data["documents"]]) # Sem persistir, poucos dados
    return index, data["ids"], data["documents"], data["metadatas"]

# Busca Lexica - Sparse

def bm25_search(query: str, k: int = TOP_K) -> list[dict]:
    """Busca léxica: os k documentos com mais termos exatos da pergunta (maior score = mais relevante)."""
    index, ids, docs, metas = bm25_index()
    scores = index.get_scores(tokenize(query))
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {"id": ids[i], "texto": docs[i], "metadata": metas[i], "score": float(scores[i])}
        for i in best
        if scores[i] > 0  # score 0 = nenhuma palavra em comum
    ]


RRF_K = 60  # constante de suavização, top-k muito pequeno aumenta a probabilidade de 1 lugar de uma lista ficar no top final
def rrf(*rankings: list[dict]) -> list[dict]:
    """Reciprocal Rank Fusion: soma 1/(RRF_K + posição) de cada documento em cada ranking."""
    fused = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking, start=1):
            entry = fused.setdefault(doc["id"], {"id": doc["id"], "texto": doc["texto"],
                                                 "metadata": doc["metadata"], "rrf": 0.0})
            entry["rrf"] += 1 / (RRF_K + position)
    return sorted(fused.values(), key=lambda d: d["rrf"], reverse=True)

#Busca hibrida - uso de rrf pra rankear as posicoes de top-k

# 10 candidatos pra busca ampla na fusao mantendo o topk 5 pra llm evitando maiores ruidos com contextos nao semanticos similarmente
def hybrid_search(query: str, k: int = TOP_K, candidates: int = 2 * TOP_K) -> list[dict]: 
    """Busca híbrida (densa + BM25 via RRF). Lista vazia = pergunta fora do escopo da base."""
    dense = dense_search(query, k=candidates)
    # Só a busca densa tem threshold calibrado para barrar perguntas fora do escopo, entao se a densa não encontrar nada, barra a bm25
    if not dense:
        return []
    lexical = bm25_search(query, k=candidates)
    results = rrf(dense, lexical)[:k]
    # Marca de onde veio cada documento (útil para depuração e avaliação)
    dense_ids, lexical_ids = {d["id"] for d in dense}, {d["id"] for d in lexical}
    for r in results:
        r["fontes"] = [name for name, ids in (("densa", dense_ids), ("bm25", lexical_ids)) if r["id"] in ids]
    return results
