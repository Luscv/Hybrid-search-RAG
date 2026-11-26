"""Ingestão: cada linha dos CSVs tratados vira um documento no ChromaDB."""
# LEMBRAR: cada collection do chroma é uma tabela que contem dados no formato ID, metadados, documento e embedding
import time
from pathlib import Path

import chromadb
import pandas as pd
from google.genai import types

from config import client, EMBED_MODEL

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
QUALITY_FILE = ROOT / "data" / "quality" / "qualidade_dados.csv"  # opcional: indexado se existir
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "northwind"  # ADAPTE: nome da coleção da sua base
# Tabelas usadas só pelas tools (sem chave única por linha ou sem valor para busca textual)
NOT_INDEXED = {"itens_pedido"}  # ADAPTE
BATCH_SIZE = 100  # parametro deixado pra testar o batch máximo 100 ou diminuir em caso de muitos erros e retries
EMBED_PAUSE = 60  # s entre lotes: o plano gratuito limita 100 textos/min no embedding (cada texto do lote conta)


def load_tables() -> dict[str, pd.DataFrame]:
    """Lê cada CSV tratado e o relatório de qualidade; o nome do arquivo vira o nome da tabela."""
    # transformar tudo em texto para dados como telefone e CEP nao serem tratados como float
    # perde o tratamento numérico mas questões analíticas serão retornadas por tools parametrizadas
    tables = {csv.stem: pd.read_csv(csv, dtype=str) for csv in sorted(PROCESSED_DIR.glob("*.csv"))}
    if QUALITY_FILE.exists():  # o relatório de qualidade também vira documento consultável
        tables["qualidade_dados"] = pd.read_csv(QUALITY_FILE, dtype=str)
    return tables


def enrich(tables: dict[str, pd.DataFrame]) -> None:
    """ADAPTE: desnormaliza as tabelas, copiando nomes das tabelas relacionadas para cada linha.

    O RAG não faz join: se o pedido só tiver "id_cliente: QUICK", a pergunta "pedidos da QUICK-Stop"
    não encontra nada. Cada documento precisa ser autoexplicativo.
    """
    clientes = tables["clientes"][["id_cliente", "empresa"]].rename(columns={"empresa": "cliente"})
    funcionarios = tables["funcionarios"][["id_funcionario", "nome"]]
    transportadoras = tables["transportadoras"][["id_transportadora", "empresa"]].rename(
        columns={"empresa": "transportadora"})
    produtos = tables["produtos"][["id_produto", "produto"]]

    # Itens do pedido resumidos numa coluna de texto: "Queso Cabrales x12; Singaporean Hokkien Fried Mee x10"
    itens = tables["itens_pedido"].merge(produtos, on="id_produto", how="left")
    itens["item"] = itens["produto"] + " x" + itens["quantidade"]
    itens = itens.groupby("id_pedido", sort=False)["item"].agg("; ".join).rename("itens").reset_index()

    # left-join mantém pedidos sem cliente/funcionário/itens correspondentes
    tables["pedidos"] = (tables["pedidos"]
                         .merge(clientes, on="id_cliente", how="left")
                         .merge(funcionarios.rename(columns={"nome": "funcionario"}), on="id_funcionario", how="left")
                         .merge(transportadoras, on="id_transportadora", how="left")
                         .merge(itens, on="id_pedido", how="left"))

    categorias = tables["categorias"][["id_categoria", "categoria"]]
    fornecedores = tables["fornecedores"][["id_fornecedor", "empresa"]].rename(columns={"empresa": "fornecedor"})
    tables["produtos"] = (tables["produtos"]
                          .merge(categorias, on="id_categoria", how="left")
                          .merge(fornecedores, on="id_fornecedor", how="left"))

    supervisores = funcionarios.rename(columns={"id_funcionario": "id_supervisor", "nome": "nome_supervisor"})
    tables["funcionarios"] = tables["funcionarios"].merge(supervisores, on="id_supervisor", how="left")


def row_to_text(table: str, row: pd.Series) -> str:
    """Ex.: 'Tabela: produtos | id_produto: 38 | produto: Côte de Blaye | ...'"""
    fields = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]  # pular nulos pra evitar ruído no embedding
    return f"Tabela: {table} | " + " | ".join(fields)


def build_documents(tables: dict[str, pd.DataFrame]) -> tuple[list, list, list]:
    """Gera (ids, textos, metadatas). O ID usa a chave da tabela: 'pedidos:10248'."""
    # util para a busca semantica e lexica compreender o valor referencia ex: 12 vira quantidade:12 dando sentido semantico/lexico ao valor
    ids, docs, metas = [], [], []
    for table, df in tables.items():
        if table in NOT_INDEXED:
            continue
        key = df.columns[0]  # convenção: a 1ª coluna de cada CSV é a chave, o que garante rastreabilidade da fonte
        assert df[key].is_unique, f"{table}: chave {key} duplicada"
        for _, row in df.iterrows():
            ids.append(f"{table}:{row[key]}")
            docs.append(row_to_text(table, row))
            # Chroma não aceita valores nulos na metadata
            metas.append({"tabela": table, **{col: val for col, val in row.items() if pd.notna(val)}})
    return ids, docs, metas


def embed(texts: list[str], task_type: str) -> list[list[float]]:  # mesma funcao de embed pra pergunta a fim de manter um padrao
    """Embeda em lotes. task_type: RETRIEVAL_DOCUMENT (ingestão) ou RETRIEVAL_QUERY (pergunta)."""
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        if start > 0:
            time.sleep(EMBED_PAUSE)  # espera a cota do minuto renovar antes do próximo lote
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[start:start + BATCH_SIZE],
            config=types.EmbedContentConfig(task_type=task_type),  # definir se é embed de query ou do documento
        )
        vectors.extend(e.values for e in response.embeddings)
    return vectors


def get_collection() -> chromadb.Collection:
    """Abre (ou cria) a coleção. Usada também pelo retrieval."""
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))  # Persistencia em disco, evitar re-embeddings
    return db.get_or_create_collection(
        COLLECTION,
        embedding_function=None,  # os vetores vêm sempre do Gemini (mesmo ecossistema da LLM free e suporte multilingual)
        configuration={"hnsw": {"space": "cosine"}},  # busca semantica por direção, apesar de ja normalizado no modelo gemini
    )


def main() -> None:
    tables = load_tables()
    enrich(tables)
    ids, docs, metas = build_documents(tables)
    n_batches = -(-len(docs) // BATCH_SIZE)
    print(f"{len(docs)} documentos de {len(tables) - len(NOT_INDEXED)} tabelas "
          f"({n_batches} lotes, ~{(n_batches - 1) * EMBED_PAUSE // 60} min por causa da pausa entre lotes)")

    vectors = embed(docs, "RETRIEVAL_DOCUMENT")

    # Recria a coleção do zero: rodar de novo sempre gera o mesmo índice, sem documentos órfãos
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION in [c.name for c in db.list_collections()]:
        db.delete_collection(COLLECTION)
    collection = get_collection()
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)

    assert collection.count() == len(ids), "nem todos os documentos foram gravados"
    print(f"Coleção '{COLLECTION}' com {collection.count()} documentos em {CHROMA_DIR}")


if __name__ == "__main__":
    main()
