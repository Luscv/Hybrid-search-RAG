"""Tools analíticas: cálculos determinísticos em pandas sobre a base completa.

O LLM não calcula: ele só escolhe a função e preenche os parâmetros. Toda conta é feita aqui.

ADAPTE: este arquivo é o mais ligado ao domínio. Para outra base, mantenha o padrão
(validação → filtro → agregação → ressalvas + citacao + criterio) e troque tabelas e colunas.
"""
from functools import lru_cache

import pandas as pd

from ingest import PROCESSED_DIR

METRICAS = {"faturamento": "valor", "volume": "quantidade"}
AGRUPAMENTOS_RANKING = ("funcionario", "cliente", "pais", "produto", "categoria")
AGRUPAMENTOS_COMPARACAO = ("pais", "funcionario", "produto", "categoria")
CRITERIO = (
    "Faturamento = soma do valor líquido dos itens (preço praticado × quantidade × (1 − desconto)), "
    "sem frete; volume = soma das unidades. Entram todos os pedidos, inclusive os ainda não enviados. "
    "País = país de entrega do pedido."
)

# funcoes auxiliares


@lru_cache(maxsize=1)
def itens() -> pd.DataFrame:
    """Itens de pedido com ano, país e nomes de funcionário, cliente, produto e categoria."""
    itens = pd.read_csv(PROCESSED_DIR / "itens_pedido.csv")
    pedidos = pd.read_csv(PROCESSED_DIR / "pedidos.csv",
                          usecols=["id_pedido", "id_cliente", "id_funcionario", "ano_mes", "pais_entrega"])
    clientes = pd.read_csv(PROCESSED_DIR / "clientes.csv", usecols=["id_cliente", "empresa"])
    funcionarios = pd.read_csv(PROCESSED_DIR / "funcionarios.csv", usecols=["id_funcionario", "nome"])
    produtos = pd.read_csv(PROCESSED_DIR / "produtos.csv", usecols=["id_produto", "produto", "id_categoria"])
    categorias = pd.read_csv(PROCESSED_DIR / "categorias.csv", usecols=["id_categoria", "categoria"])

    df = (itens.merge(pedidos, on="id_pedido", how="left")
          .merge(clientes, on="id_cliente", how="left")
          .merge(funcionarios, on="id_funcionario", how="left")
          .merge(produtos, on="id_produto", how="left")
          .merge(categorias, on="id_categoria", how="left"))
    df["ano"] = df["ano_mes"].str[:4].astype(int)
    df["pais"] = df["pais_entrega"]
    # nome + ID: deixa o rótulo inequívoco e citável
    df["funcionario"] = df["nome"] + " (" + df["id_funcionario"].astype(str) + ")"
    df["cliente"] = df["empresa"] + " (" + df["id_cliente"] + ")"
    df["produto"] = df["produto"] + " (" + df["id_produto"].astype(str) + ")"
    return df


@lru_cache(maxsize=1)
def meses_por_ano() -> dict[int, list[str]]:
    """Meses com pedidos em cada ano: revela anos parciais (ex.: 1996 começa em julho)."""
    df = itens()
    return {int(ano): sorted(g["ano_mes"].unique()) for ano, g in df.groupby("ano")}


def _anos_parciais(*anos: int) -> list[str]:
    """Ressalva para anos com menos de 12 meses de dados (comparação anual distorcida)."""
    ressalvas = []
    for ano in anos:
        meses = meses_por_ano().get(ano, [])
        if 0 < len(meses) < 12:
            ressalvas.append(f"{ano} é um ano parcial: só há pedidos de {meses[0]} a {meses[-1]} "
                             f"({len(meses)} meses).")
    return ressalvas


def _validar(
    ano: int | None = None,
    pais: str | None = None,
    metrica: str | None = None,
    por: str | None = None,
    opcoes_por: tuple = (),
) -> str | None:
    """Devolve uma mensagem de erro se algum parâmetro for inválido; None se estiver tudo certo."""
    if metrica is not None and metrica not in METRICAS:
        return f"Métrica '{metrica}' inválida. Opções: {', '.join(METRICAS)}."
    if por is not None and por not in opcoes_por:
        return f"Agrupamento '{por}' inválido. Opções: {', '.join(opcoes_por)}."
    anos = sorted(meses_por_ano())
    if ano is not None and ano not in anos:
        return f"Ano {ano} fora do período da base ({anos[0]} a {anos[-1]})."
    paises = sorted(itens()["pais"].dropna().unique())
    if pais is not None and pais.lower() not in [p.lower() for p in paises]:
        return f"País '{pais}' sem pedidos. Países disponíveis (em inglês): {', '.join(paises)}."
    return None


def _filtrar_nome(df: pd.DataFrame, coluna: str, valor: str) -> tuple[pd.DataFrame, str | None]:
    """Filtra por nome ou ID parcial; erro se o termo bater com nenhum ou com mais de um."""
    df = df[df[coluna].str.contains(valor, case=False, regex=False)]
    encontrados = sorted(df[coluna].unique())
    if len(encontrados) != 1:  # nenhum ou mais de um: não somar entidades diferentes sem avisar
        return df, f"{coluna.capitalize()} '{valor}' ambíguo ou sem pedidos. Encontrados: {encontrados}."
    return df, None

# 3 tools analiticas: soma, ranking e comparação


def consultar_total(
    metrica: str = "faturamento",
    ano: int | None = None,
    pais: str | None = None,
    funcionario: str | None = None,
    cliente: str | None = None,
) -> dict:
    """Total de faturamento (US$) ou volume (unidades) dos pedidos, com filtros opcionais.

    Use para perguntas como "qual o faturamento de 1997?" ou "quanto a Margaret vendeu para a Alemanha?".
    Não use para ordenar grupos (use ranking) nem para comparar anos (use comparar_anos).
    metrica: "faturamento" (US$) ou "volume" (unidades). ano: entre 1996 e 1998.
    pais: nome em inglês (ex.: Germany, USA, UK). funcionario: nome ou ID (ex.: Margaret ou 4).
    cliente: empresa ou ID (ex.: QUICK-Stop ou QUICK).
    Se retornar erro de nome ambíguo, liste os candidatos e peça ao usuário para escolher.
    O resultado traz "citacao" (copie na resposta) e "ressalvas" (mencione todas).
    """
    if erro := _validar(ano, pais, metrica):
        return {"erro": erro}
    df = itens()
    if ano is not None:
        df = df[df["ano"] == ano]
    if pais is not None:
        df = df[df["pais"].str.lower() == pais.lower()]
    for coluna, valor in (("funcionario", funcionario), ("cliente", cliente)):
        if valor is not None:
            df, erro = _filtrar_nome(df, coluna, valor)
            if erro:
                return {"erro": erro}
    ressalvas = _anos_parciais(ano) if ano is not None else []
    if len(df) == 0:
        ressalvas.append("Nenhum pedido com esses filtros: isso não significa faturamento zero.")
    return {
        "metrica": metrica,
        "valor": round(float(df[METRICAS[metrica]].sum()), 2),
        "n_pedidos": int(df["id_pedido"].nunique()),
        "filtros": {"ano": ano, "pais": pais, "funcionario": funcionario, "cliente": cliente},
        "ressalvas": ressalvas,
        "citacao": "[ferramenta:consultar_total]",
        "criterio": CRITERIO,
    }


def ranking(
    por: str,
    metrica: str = "faturamento",
    ano: int | None = None,
    n: int = 3,
) -> dict:
    """Top N funcionários, clientes, países, produtos ou categorias por faturamento (US$) ou volume (unidades).

    Use para perguntas como "quem são os top 3 clientes por volume?" ou "qual o melhor vendedor?".
    Não use para o total de um único filtro (use consultar_total).
    por: "funcionario", "cliente", "pais", "produto" ou "categoria". metrica: "faturamento" ou "volume".
    ano: entre 1996 e 1998. Omitir = período todo (campo "periodo"). n: quantidade de posições (1 a 20).
    O resultado traz "citacao" (copie na resposta), "periodo" e "ressalvas" (mencione todas).
    """
    if erro := _validar(ano, metrica=metrica, por=por, opcoes_por=AGRUPAMENTOS_RANKING):
        return {"erro": erro}
    df = itens()
    if ano is not None:
        df = df[df["ano"] == ano]
    grupos = (df.groupby(por)
              .agg(total=(METRICAS[metrica], "sum"), n_pedidos=("id_pedido", "nunique"))
              .sort_values("total", ascending=False))
    n = max(1, min(n, 20))
    top = grupos.head(n)
    ressalvas = _anos_parciais(ano) if ano is not None else []
    if len(grupos) == 0:
        ressalvas.append("Nenhum pedido no período: isso não significa faturamento zero.")
    elif len(grupos) < n:
        ressalvas.append(f"Só {len(grupos)} grupo(s) com pedidos; o ranking tem menos de {n} posições.")
    periodo = f"{df['ano_mes'].min()} a {df['ano_mes'].max()}" if len(df) else None  # período que a conta cobriu
    return {
        "metrica": metrica,
        "por": por,
        "ano": ano,
        "periodo": periodo,
        "ranking": [
            {"posicao": i, por: nome, "valor": round(float(row["total"]), 2), "n_pedidos": int(row["n_pedidos"])}
            for i, (nome, row) in enumerate(top.iterrows(), start=1)
        ],
        "ressalvas": ressalvas,
        "citacao": "[ferramenta:ranking]",
        "criterio": CRITERIO,
    }


def comparar_anos(
    ano_inicial: int,
    ano_final: int,
    por: str = "pais",
    metrica: str = "faturamento",
) -> dict:
    """Compara faturamento ou volume entre dois anos, por país, funcionário, produto ou categoria.

    Use para perguntas como "quais países tiveram queda de vendas em 1997?" (comparar 1996 com 1997).
    Não use para um ano isolado (use consultar_total ou ranking).
    por: "pais", "funcionario", "produto" ou "categoria". metrica: "faturamento" ou "volume".
    Grupos sem pedidos em um dos anos não têm variação calculada: ausência de registro não é zero.
    Esses grupos vêm em "grupos_sem_dados". Anos parciais distorcem a comparação e vêm em "ressalvas".
    Mencione ambos na resposta. O resultado traz "citacao" (copie na resposta).
    """
    if erro := _validar(metrica=metrica, por=por, opcoes_por=AGRUPAMENTOS_COMPARACAO):
        return {"erro": erro}
    for ano in (ano_inicial, ano_final):
        if erro := _validar(ano):
            return {"erro": erro}
    df = itens()
    tabela = (df[df["ano"].isin([ano_inicial, ano_final])]
              .groupby([por, "ano"])[METRICAS[metrica]].sum()
              .unstack())  # uma coluna por ano; NaN = sem pedidos naquele ano

    resultado = []
    for nome, row in tabela.iterrows():
        antes, depois = row.get(ano_inicial), row.get(ano_final)
        if pd.isna(antes) or pd.isna(depois):
            ano_vazio = ano_inicial if pd.isna(antes) else ano_final
            situacao, variacao = f"sem pedidos em {ano_vazio}", None
        else:
            variacao = round(float((depois / antes - 1) * 100), 2)
            situacao = "queda" if variacao < 0 else "alta" if variacao > 0 else "estável"
        resultado.append({
            por: nome,
            str(ano_inicial): None if pd.isna(antes) else round(float(antes), 2),
            str(ano_final): None if pd.isna(depois) else round(float(depois), 2),
            "variacao_pct": variacao,
            "situacao": situacao,
        })
    sem_dados = [r[por] for r in resultado if r["variacao_pct"] is None]
    ressalvas = _anos_parciais(ano_inicial, ano_final)
    if sem_dados:
        ressalvas.append(f"{', '.join(map(str, sem_dados))}: sem pedidos em um dos anos, variação não calculada "
                         "(não é queda de 100%).")
    return {
        "metrica": metrica,
        "por": por,
        "comparacao": resultado,
        "grupos_sem_dados": sem_dados,
        "ressalvas": ressalvas,
        "citacao": "[ferramenta:comparar_anos]",
        "criterio": CRITERIO,
    }


# Lista entregue ao LLM no generate.py
TOOLS = [consultar_total, ranking, comparar_anos]


if __name__ == "__main__":
    # Autoteste: os números precisam bater com o gabarito (eval/gabarito.ipynb), calculado sem este arquivo
    melhor = ranking("funcionario", n=1)["ranking"][0]
    assert melhor["funcionario"] == "Margaret Peacock (4)" and melhor["valor"] == 232890.85, melhor
    top3 = [r["cliente"] for r in ranking("cliente", metrica="volume")["ranking"]]
    assert [c[-6:-1] for c in top3] == ["SAVEA", "ERNSH", "QUICK"], top3
    comparacao = comparar_anos(1997, 1998)
    assert any("1998 é um ano parcial" in r for r in comparacao["ressalvas"]), comparacao["ressalvas"]
    assert "Argentina" in comparar_anos(1996, 1997)["grupos_sem_dados"]  # sem pedidos em 1996, não é alta infinita
    assert "erro" in consultar_total(funcionario="an")  # Nancy, Andrew, Janet, Anne...
    assert "erro" in consultar_total(ano=2030)
    print("Autoteste ok.\n")

    for nome, resultado in [
        ("consultar_total(ano=1997)", consultar_total(ano=1997)),
        ("consultar_total(funcionario='Margaret', ano=1997)", consultar_total(funcionario="Margaret", ano=1997)),
        ("ranking('cliente', 'volume')", ranking("cliente", "volume")),
        ("comparar_anos(1997, 1998)", comparar_anos(1997, 1998)),
        ("consultar_total(pais='Brasil')", consultar_total(pais="Brasil")),
    ]:
        resultado.pop("criterio", None)
        print(f"{nome}\n  {resultado}\n")
