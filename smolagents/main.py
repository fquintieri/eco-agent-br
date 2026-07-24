import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv
from smolagents import tool, ToolCallingAgent, OpenAIServerModel

"""
==============================================================================
EcoAgent BR - Módulo Smolagents (Hugging Face)
Recursos: Guardrail de Escopo + Context Truncation + Validação Pós-Execução
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

MAX_HISTORY_STEPS = 6

# ==============================================================================
# PROMPT DO SISTEMA (MASTER SKILL PADRONIZADA)
# ==============================================================================
SYSTEM_PROMPT = """
Você é o **EcoAgent BR**, um analista macroeconômico sênior especializado EXCLUSIVAMENTE em indicadores do Banco Central do Brasil e finanças.

1. TRAVA DE ESCOPO E SEGURANÇA (GUARDRAIL ABSOLUTO):
   - Seu domínio de atuação é ESTRITAMENTE: economia brasileira, taxa SELIC, IPCA, Dólar, mercado financeiro e investimentos.
   - Se o usuário perguntar sobre assuntos FORA desse domínio (ex: esportes, futebol, animais, receitas, curiosidades):
     * NÃO chame nenhuma ferramenta e recuse com: "Como sou o EcoAgent BR, meu foco é exclusivamente a análise econômica do Brasil e indicadores do Banco Central. Como posso te ajudar com a taxa SELIC, IPCA, Dólar ou estratégias de investimentos hoje?"

2. SAUDAÇÕES E IDENTIDADE:
   - Para 'oi', 'olá' ou 'quem é você?': NÃO chame ferramentas. Responda em 1 frase curta, apresentando-se como EcoAgent BR.

3. CONSULTA PONTUAL DE UM INDICADOR (ex: "Qual a SELIC?", "Cotação do Dólar"):
   - PASSO 1: chame APENAS a ferramenta do indicador pedido e aguarde o retorno.
   - PASSO 2: só então chame final_answer com 1 parágrafo direto contendo o valor e a data. NÃO monte tabelas.

4. RELATÓRIOS E PANORAMAS (ex: recomendações, análises, "onde investir?"):
   - Acione TODAS as 3 ferramentas (`get_selic_rate`, `get_ipca_rate` e `get_usd_exchange_rate`) e só então responda.
   - Estruture o parâmetro 'answer' do final_answer em Markdown:
     ### 📌 Resumo Executivo
     (Leitura do cenário com base nas métricas coletadas.)

     ### 📊 Tabela de Indicadores Econômicos
     (Tabela Markdown com: Indicador, Valor Atual e Data de Referência com valores reais.)

     ### 💡 Análise de Impacto e Estratégia de Investimentos
     - **Cálculo do Juro Real:** explicite (Juro Real ≈ SELIC - IPCA).
     - **Renda Fixa vs. Variável:** comente a atratividade de cada classe.

REGRA CRÍTICA DE TOOL CALLING: nunca escreva chamadas de função, código ou placeholders no texto final. Primeiro chame as ferramentas necessárias em um passo; depois, em outro passo, invoque `final_answer` com o texto devidamente preenchido. Use SOMENTE os valores retornados pelas ferramentas.
"""

# Wrappers de ferramentas para o Smolagents
selic_tool = tool(get_selic_rate)
ipca_tool = tool(get_ipca_rate)
usd_tool = tool(get_usd_exchange_rate)


# ==============================================================================
# CONTEXT TRUNCATION E VALIDAÇÃO DE RESPOSTA
# ==============================================================================
def truncar_memoria_do_agente(agent: ToolCallingAgent, max_steps: int = 6):
    """
    Trunca a lista de passos armazenados na memória interna do Smolagents
    (agent.memory.steps), evitando estourar a janela de contexto de tokens.
    """
    if hasattr(agent, "memory") and hasattr(agent.memory, "steps"):
        steps = agent.memory.steps
        if len(steps) > max_steps:
            # Mantém apenas os últimos max_steps na memória ativa
            agent.memory.steps = steps[-max_steps:]


def validar_resposta_do_agente(resposta: str, consulta_economica: bool = False) -> Dict[str, Any]:
    """Inspeciona a resposta para garantir formatação e qualidade."""
    erros = []

    if not resposta or len(resposta.strip()) < 10:
        erros.append("Resposta gerada está vazia ou excessivamente curta.")

    # Se a resposta for uma recusa de escopo ou saudação, ignora a validação de estrutura de relatório
    eh_recusa_ou_saudacao = any(p in resposta.lower() for p in ["não posso", "meu foco é", "exclusivamente", "como posso ajudar"])

    if consulta_economica and not eh_recusa_ou_saudacao:
        secoes_obrigatorias = ["Resumo Executivo", "Tabela de Indicadores", "Análise de Impacto"]
        for secao in secoes_obrigatorias:
            if secao.lower() not in resposta.lower():
                erros.append(f"Seção obrigatória ausente no relatório: '{secao}'")

        if "|" not in resposta:
            erros.append("A resposta econômica não contém uma tabela Markdown formatada.")

    return {
        "valido": len(erros) == 0,
        "erros": erros
    }


def criar_agente_smolagents() -> ToolCallingAgent:
    """Configura o modelo e instancia o ToolCallingAgent."""
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    # Sanitiza caso o valor no .env venha com o prefixo 'llm_model='
    if "llm_model=" in model_name.lower():
        model_name = model_name.split("=")[-1]

    model = OpenAIServerModel(
        model_id=model_name,
        api_base=base_url,
        api_key=api_key
    )

    return ToolCallingAgent(
        tools=[selic_tool, ipca_tool, usd_tool],
        model=model,
        instructions=SYSTEM_PROMPT
    )


# ==============================================================================
# CHAT INTERATIVO (CLI)
# ==============================================================================
if __name__ == "__main__":
    print("==================================================")
    print("  EcoAgent BR (Smolagents) - Chat Interativo Otimizado")
    print("  Recursos: Guardrail + Truncation Corrigido + Validação")
    print("==================================================")

    agent = criar_agente_smolagents()

    while True:
        try:
            entrada_usuario = input("\nVocê: ").strip()

            if entrada_usuario.lower() in ["sair", "exit", "quit"]:
                print("Encerrando a sessão do EcoAgent BR (Smolagents). Até logo!")
                break

            if not entrada_usuario:
                continue

            # 1. Trunca a memória de passos antes da execução
            truncar_memoria_do_agente(agent, max_steps=MAX_HISTORY_STEPS)

            # 2. Executa a chamada do agente mantendo o estado da sessão (reset=False)
            resposta = agent.run(entrada_usuario, reset=False)

            # 3. Validação pós-execução
            palavras_chave_relatorio = ["investimento", "relatorio", "panorama", "melhor", "estrategia"]
            eh_consulta_economica = any(p in entrada_usuario.lower() for p in palavras_chave_relatorio)

            validacao = validar_resposta_do_agente(str(resposta), consulta_economica=eh_consulta_economica)

            if not validacao["valido"]:
                print("\n[ALERTA DE VALIDAÇÃO]: A resposta gerada apresentou inconsistências:")
                for erro in validacao["erros"]:
                    print(f"  - ⚠️ {erro}")
                print("--------------------------------------------------")

            print("\nRESPOSTA FINAL DO AGENTE:")
            print(resposta)
            print("=" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nSessão encerrada pelo usuário.")
            break
        except Exception as erro:
            print(f"\n[ERRO CRÍTICO] Falha na execução do Smolagents: {erro}")