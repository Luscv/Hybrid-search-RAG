"""Calibração do threshold de similaridade.

Compara a distância do melhor documento para perguntas DENTRO e FORA do escopo da base.
O threshold deve ficar entre as duas faixas. Consulta o Chroma direto (sem filtro de
distância), pois a calibração precisa ver as distâncias brutas.
"""
from pathlib import Path

import pandas as pd

from ingest import ROOT, embed
from retrieval import open_collection

OUTPUT_FILE = ROOT / "eval" / "calibracao_threshold.csv"

# ADAPTE: ~10 perguntas de cada lado, escritas como um usuário real escreveria
IN_SCOPE = [
    "os dados estão prontos para uso?",
    "havia problemas de formatação nos arquivos originais?",
    "pedidos da QUICK-Stop",
    "pedido 10250",
    "quem é o supervisor do Robert King?",
    "qual o telefone da Alfreds Futterkiste?",
    "quais produtos estão sem estoque?",
    "quem fornece o Chai?",
    "qual categoria tem frutos do mar?",
    "pedidos enviados pela Speedy Express para o Brasil",
    # perguntas analíticas no escopo, mesmo que a resposta venha de tools
    "quais países tiveram queda de vendas em 1998?",
    "quem são os top 3 clientes por volume?",
]

OUT_OF_SCOPE = [
    # sem relação com a base
    "qual a capital da França?",
    "receita de bolo de cenoura",
    "quem ganhou a Copa do Mundo de 2022?",
    "como funciona a fotossíntese?",
    "previsão do tempo para amanhã em São Paulo",
    # mesmo domínio (alimentos/empresa), mas a informação não está na base
    "qual o preço do Chai na concorrência?",
    "qual o salário da Nancy Davolio?",
    "qual a validade do queijo Gorgonzola?",
    "qual a política de devolução dos produtos?",
    "qual o horário de funcionamento do depósito?",
]


def best_matches(questions: list[str]) -> pd.DataFrame:
    """Para cada pergunta, o documento mais próximo e sua distância (1 chamada de embedding para todas)."""
    vectors = embed(questions, "RETRIEVAL_QUERY")
    res = open_collection().query(query_embeddings=vectors, n_results=1)
    return pd.DataFrame({
        "pergunta": questions,
        "melhor_documento": [ids[0] for ids in res["ids"]],
        "distancia": [dists[0] for dists in res["distances"]],
    })


def main() -> None:
    df = best_matches(IN_SCOPE + OUT_OF_SCOPE)
    df.insert(0, "escopo", ["dentro"] * len(IN_SCOPE) + ["fora"] * len(OUT_OF_SCOPE))
    df = df.sort_values("distancia")

    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format="%.3f"))

    max_in = df.loc[df["escopo"] == "dentro", "distancia"].max()
    min_out = df.loc[df["escopo"] == "fora", "distancia"].min()
    print(f"\nPior distância DENTRO do escopo: {max_in:.3f}")
    print(f"Melhor distância FORA do escopo: {min_out:.3f}")
    if max_in < min_out:
        print(f"Faixas separadas. Threshold sugerido (ponto médio): {(max_in + min_out) / 2:.3f}")
    else:
        print("Faixas SOBREPOSTAS: nenhum threshold separa todas as perguntas — ver casos na fronteira.")

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResultados salvos em {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
