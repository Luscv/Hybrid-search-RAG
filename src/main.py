"""Teste ao vivo: loop de perguntas no terminal, com rastreio de fontes, tools e citações.

Cada pergunta consome 1 requisição de geração (2 quando usa uma tool), e perguntas barradas
pelo portão de escopo não chamam o LLM. As interações ficam em eval/logs/sessao_<data>.jsonl.
"""
import json
from datetime import datetime

from google.genai import errors

from config import LLM_MODEL
from generate import answer
from ingest import ROOT
from retrieval import bm25_index

LOG_DIR = ROOT / "eval" / "logs"
EXIT_COMMANDS = {"sair", "exit", "quit"}


def show(result: dict) -> None:
    """Imprime a resposta e o rastreio: tools (args e ressalvas), documentos (origem e RRF) e citações inválidas."""
    print(f"\n{result['resposta']}\n")
    for call in result["tools"]:
        print(f"  tool: {call['tool']}({call['args']})")
        resultado = (call.get("resultado") or {}).get("result", {})  # o SDK embrulha o retorno em {"result": ...}
        if resultado.get("erro"):
            print(f"    erro: {resultado['erro']}")
        for ressalva in resultado.get("ressalvas", []):
            print(f"    ressalva: {ressalva}")
    for doc in result["docs"]:
        print(f"  doc: {doc['id']:<28} origem={'+'.join(doc['origem']):<11} rrf={doc['rrf']}")
    if not result["docs"]:
        print("  (fora do escopo: nenhum documento passou no threshold, LLM não chamado)")
    if result["citacoes_invalidas"]:
        print(f"  ⚠ citações inválidas: {result['citacoes_invalidas']}")


def log(file, question: str, result: dict) -> None:
    """Grava uma linha JSON por interação, para rever depois o que o sistema recebeu e respondeu."""
    entry = {"hora": datetime.now().isoformat(timespec="seconds"), "modelo": LLM_MODEL,
             "pergunta": question, **result}
    file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    file.flush()  # não perde a linha se a sessão for interrompida


def main() -> None:
    bm25_index()  # monta o índice antes do loop: a primeira pergunta não fica mais lenta
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"sessao_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    print(f"RAG híbrido ({LLM_MODEL}). Digite uma pergunta ou 'sair'. Log: {log_file.relative_to(ROOT)}")

    with log_file.open("a", encoding="utf-8") as file:
        while True:
            try:
                question = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):  # Ctrl+C / Ctrl+D encerram sem traceback
                break
            if not question:
                continue
            if question.lower() in EXIT_COMMANDS:
                break
            try:
                result = answer(question)
            except errors.APIError as e:  # 503/504/429 do plano gratuito: avisa e continua a sessão
                print(f"\nErro da API do Gemini ({e.code}): {e.message}")
                continue
            except KeyboardInterrupt:  # interrompe só a pergunta em andamento
                print("\nPergunta cancelada.")
                continue
            show(result)
            log(file, question, result)
    print("\nAté mais.")


if __name__ == "__main__":
    main()
