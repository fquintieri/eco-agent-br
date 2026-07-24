import os
import sys

# A telemetria do CrewAI tenta exportar spans via OTLP e polui o console com
# warnings quando não há coletor. Desligamos ANTES de importar o pacote.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from typing import Dict, Any, Type, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Dependências do CrewAI
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

"""
==============================================================================
EcoAgent BR - Módulo CrewAI (orientado a Papéis e Tarefas)
Recursos: Subclasse BaseTool + Pydantic + Memória Multi-turno + Validador
==============================================================================
"""

# No Windows o console padrão usa cp1252 e quebra acentos/emojis; forçamos UTF-8.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.bcb_tools import (
    get_selic_rate,
    get_ipca_rate,
    get_usd_exchange_rate
)

load_dotenv()


# ==============================================================================
# 1. SCHEMAS E FERRAMENTAS DO CREWAI
# ==============================================================================
class ConsultaInput(BaseModel):
    consulta: str = Field(
        default="dados",
        description="Parâmetro opcional de consulta para os indicadores do Banco Central."
    )


class SelicTool(BaseTool):
    name: str = "Obter Taxa SELIC Meta"
    description: str = "Consulta a taxa SELIC meta atualizada no Banco Central do Brasil."
    args_schema: Type[BaseModel] = ConsultaInput

    def _run(self, consulta: str = "dados") -> str:
        return get_selic_rate()


class IpcaTool(BaseTool):
    name: str = "Obter Inflação IPCA"
    description: str = "Consulta o índice de inflação mensal (IPCA) mais recente no Banco Central do Brasil."
    args_schema: Type[BaseModel] = ConsultaInput

    def _run(self, consulta: str = "dados") -> str:
        return get_ipca_rate()


class UsdTool(BaseTool):
    name: str = "Obter Cotação do Dólar PTAX"
    description: str = "Consulta a cotação oficial do Dólar comercial (PTAX) no Banco Central do Brasil."
    args_schema: Type[BaseModel] = ConsultaInput

    def _run(self, consulta: str = "dados") -> str:
        return get_usd_exchange_rate()


selic_tool = SelicTool()
ipca_tool = IpcaTool()
usd_tool = UsdTool()


# ==============================================================================
# 2. CAMADA DE VALIDAÇÃO INTELIGENTE
# ==============================================================================
def validar_resposta_do_agente(resposta: str, eh_relatorio_completo: bool = False) -> Dict[str, Any]:
    """Inspeciona a resposta gerada para garantir qualidade e integridade do relatório."""
    erros = []

    if not resposta or len(resposta.strip()) < 10:
        erros.append("Resposta gerada está vazia ou excessivamente curta.")

    # Se a resposta for recusa de escopo ou saudação, ignora validações de relatório
    eh_recusa_ou_saudacao = any(p in resposta.lower() for p in ["não posso", "meu foco é", "exclusivamente", "como posso ajudar"])

    if eh_relatorio_completo and not eh_recusa_ou_saudacao:
        secoes_obrigatorias = ["Resumo Executivo", "Tabela de Indicadores", "Análise de Impacto"]
        for secao in secoes_obrigatorias:
            if secao.lower() not in resposta.lower():
                erros.append(f"Seção obrigatória ausente no relatório: '{secao}'")

        if "|" not in resposta:
            erros.append("O relatório econômico não contém uma tabela Markdown formatada.")

    return {
        "valido": len(erros) == 0,
        "erros": erros
    }


# ==============================================================================
# 3. CONSTRUÇÃO DO AGENTE E EXECUÇÃO
# ==============================================================================
def executar_agente_crewai(entrada_usuario: str, historico: List[Dict[str, str]] = None) -> str:
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    llm = LLM(
        model=f"openai/{model_name}",
        base_url=base_url,
        api_key=api_key,
        temperature=0.0
    )

    analista_agente = Agent(
        role="Analista Macroeconômico Sênior",
        goal="Fornecer respostas extremamente precisas, adaptando o formato entre respostas curtas e relatórios estruturados.",
        backstory=(
            "Você é o EcoAgent BR, especialista em macroeconomia e mercado financeiro do Brasil. "
            "Seu compromisso é responder dúvidas com dados reais do Banco Central. "
            "Você respeita rigidamente o escopo financeiro/econômico do Brasil e recusa assuntos fora do domínio."
        ),
        tools=[selic_tool, ipca_tool, usd_tool],
        verbose=False,
        memory=False,
        llm=llm
    )

    texto_historico = ""
    if historico:
        texto_historico = "HISTÓRICO DA CONVERSA:\n"
        for item in historico[-2:]:
            texto_historico += f"{item['role'].upper()}: {item['content']}\n"
        texto_historico += "\n"

    instrucoes_tarefa = f"""
    {texto_historico}
    SOLICITAÇÃO DO USUÁRIO: '{entrada_usuario}'

    DIRETRIZES DE ATENDIMENTO:

    1. GUARDRAIL DE ESCOPO:
       - Se a Pergunta for fora do escopo econômico (ex: clima, esportes, receitas, futebol):
         * NÃO execute NENHUMA ferramenta.
         * Responda: "Como sou o EcoAgent BR, meu foco é exclusivamente a análise econômica do Brasil e indicadores do Banco Central. Como posso te ajudar com a taxa SELIC, IPCA, Dólar ou estratégias de investimentos hoje?"

    2. SAUDAÇÕES OU IDENTIFICAÇÃO:
       - Se for 'oi', 'olá', 'quem é você?':
         * NÃO execute NENHUMA ferramenta.
         * Responda em 1 frase curta se apresentando amigavelmente como EcoAgent BR.

    3. CONSULTAS PONTUAIS DE INDICADOR:
       - Se for 'qual o dólar?', 'qual a selic?', 'quanto tá o ipca?':
         * Execute APENAS a ferramenta do indicador solicitado.
         * Responda em 1 parágrafo direto com o valor e a data de referência. NÃO crie tabelas.

    4. RELATÓRIOS E ANÁLISES DE INVESTIMENTO:
       - Se for 'qual o melhor investimento?', 'me passe um panorama', 'relatório completo':
         * Execute TODAS as 3 ferramentas (`Obter Taxa SELIC Meta`, `Obter Inflação IPCA` e `Obter Cotação do Dólar PTAX`).
         * Estruture o relatório completo em Markdown com:
           ###  Resumo Executivo
           ###  Tabela de Indicadores Econômicos
           ###  Análise de Impacto e Estratégia de Investimentos (incluindo cálculo de Juro Real ≈ SELIC - IPCA)
    """

    formato_esperado = "Resposta textual ou relatório formatado rigorosamente de acordo com a diretriz correspondente."

    tarefa_analise = Task(
        description=instrucoes_tarefa,
        expected_output=formato_esperado,
        agent=analista_agente
    )

    equipe = Crew(
        agents=[analista_agente],
        tasks=[tarefa_analise],
        process=Process.sequential,
        verbose=False
    )

    resultado = equipe.kickoff()
    return str(resultado)


# ==============================================================================
# 4. CHAT INTERATIVO (CLI)
# ==============================================================================
if __name__ == "__main__":
    print("==================================================")
    print("  EcoAgent BR (CrewAI) - Chat Interativo (70B)")
    print("  Estudo de Caso: Agente Baseado em Papéis Sênior")
    print("  Digite 'sair', 'exit' ou 'quit' para encerrar.")
    print("==================================================")

    historico_sessao: List[Dict[str, str]] = []

    while True:
        try:
            entrada_usuario = input("\nVocê: ").strip()

            if entrada_usuario.lower() in ["sair", "exit", "quit"]:
                print("Encerrando a sessão do EcoAgent BR (CrewAI). Até logo!")
                break

            if not entrada_usuario:
                continue

            resposta = executar_agente_crewai(entrada_usuario, historico_sessao)

            historico_sessao.append({"role": "user", "content": entrada_usuario})
            historico_sessao.append({"role": "assistant", "content": resposta})

            palavras_relatorio = ["investimento", "relatorio", "panorama", "melhor", "estrategia"]
            eh_relatorio_completo = any(p in entrada_usuario.lower() for p in palavras_relatorio)

            validacao = validar_resposta_do_agente(resposta, eh_relatorio_completo=eh_relatorio_completo)

            if not validacao["valido"]:
                print("\n[ALERTA DE VALIDAÇÃO]: Inconsistências encontradas:")
                for erro in validacao["erros"]:
                    print(f"  - {erro}")
                print("--------------------------------------------------")

            print("\nRESPOSTA FINAL DO AGENTE:")
            print(resposta)
            print("=" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nSessão encerrada pelo usuário.")
            break
        except Exception as erro:
            print(f"\n[ERRO CRÍTICO] Falha na execução do CrewAI: {erro}")