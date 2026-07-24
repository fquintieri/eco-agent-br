import os
import sys
from typing import Annotated, Dict, Any, List, TypedDict
from dotenv import load_dotenv

# Dependências do LangGraph e LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

"""
==============================================================================
EcoAgent BR - Módulo LangGraph (Máquina de Estados / Grafo)
Recursos: State Machine (Nós + Arestas) + Token Saver (Janela Deslizante) + Guardrail
==============================================================================
"""

# No Windows o console padrão usa cp1252 e quebra acentos/emojis; forçamos UTF-8.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

# Coloca a raiz do projeto no sys.path para importar o pacote 'shared'.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.bcb_tools import (
    get_selic_rate,
    get_ipca_rate,
    get_usd_exchange_rate
)

load_dotenv()


# ==============================================================================
# 1. FERRAMENTAS DO LANGGRAPH (@tool)
# ==============================================================================
@tool
def selic_langgraph_tool() -> str:
    """Consulta a taxa SELIC meta atualizada no Banco Central do Brasil."""
    return get_selic_rate()

@tool
def ipca_langgraph_tool() -> str:
    """Consulta o índice de inflação mensal (IPCA) mais recente no Banco Central do Brasil."""
    return get_ipca_rate()

@tool
def usd_langgraph_tool() -> str:
    """Consulta a cotação oficial do Dólar comercial (PTAX) no Banco Central do Brasil."""
    return get_usd_exchange_rate()


ferramentas_catalogadas = [selic_langgraph_tool, ipca_langgraph_tool, usd_langgraph_tool]


# ==============================================================================
# 2. PROMPT DO SISTEMA (MASTER SKILL PADRONIZADA)
# ==============================================================================
SYSTEM_PROMPT = """
Você é o **EcoAgent BR**, um analista macroeconômico sênior especializado EXCLUSIVAMENTE em indicadores do Banco Central do Brasil e finanças.

1. TRAVA DE ESCOPO E SEGURANÇA (GUARDRAIL ABSOLUTO):
   - Seu domínio de atuação é ESTRITAMENTE: economia brasileira, taxa SELIC, IPCA, Dólar, mercado financeiro e investimentos do Brasil.
   - Para assuntos fora do escopo (ex: tempo, futebol, receitas, curiosidades), recuse com:
     "Como sou o EcoAgent BR, meu foco é exclusivamente a análise econômica do Brasil e indicadores do Banco Central. Como posso te ajudar com a taxa SELIC, IPCA, Dólar ou estratégias de investimentos hoje?"

2. SAUDAÇÕES OU IDENTIFICAÇÃO:
   - Se for 'oi', 'olá', 'quem é você?': NÃO chame ferramentas. Responda em 1 frase curta se apresentando.

3. CONSULTAS PONTUAIS DE INDICADOR:
   - Se perguntar por 1 indicador específico (ex: 'qual o dólar?'): Execute APENAS a ferramenta daquele indicador. Responda em 1 parágrafo curto.

4. RELATÓRIOS E CONSULTAS DE INVESTIMENTOS:
   - Se o usuário pedir análises, orientações de investimento ou panoramas (ex: 'onde aplicar R$ 500?'):
     * PASSO 1: Invoque TODAS as 3 ferramentas (`selic_langgraph_tool`, `ipca_langgraph_tool`, `usd_langgraph_tool`).
     * REGRA CRÍTICA: NÃO escreva nenhuma tabela ou texto com placeholders como '<function=...>' antes das ferramentas retornarem. Aguarde o retorno das ferramentas!
     * PASSO 2: Após receber os resultados das ferramentas, monte a resposta final em Markdown:
       ### 📌 Resumo Executivo
       ### 📊 Tabela de Indicadores Econômicos (com os valores numéricos reais)
       ### 💡 Análise de Impacto e Estratégia de Investimentos (incluindo cálculo do Juro Real ≈ SELIC - IPCA)
"""


# ==============================================================================
# 3. DEFINIÇÃO DO ESTADO DO GRAFO (AGENT STATE)
# ==============================================================================
class AgentState(TypedDict):
    """
    O estado do grafo é um dicionário contendo a lista de mensagens.
    O 'add_messages' garante que novas mensagens sejam anexadas à lista existente.
    """
    messages: Annotated[List[BaseMessage], add_messages]


# ==============================================================================
# 4. CONSTRUÇÃO E COMPILAÇÃO DO GRAFO DE ESTADOS
# ==============================================================================
def construir_grafo_ecoagent():
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    # Sanitiza caso o valor no .env venha com o prefixo 'llm_model='
    if "llm_model=" in model_name.lower():
        model_name = model_name.split("=")[-1]

    # Configuração do LLM via ChatOpenAI do LangChain
    llm = ChatOpenAI(
        model=model_name,
        openai_api_base=base_url,
        openai_api_key=api_key,
        temperature=0.0
    )

    # Acopla as ferramentas ao modelo
    llm_com_tools = llm.bind_tools(ferramentas_catalogadas)

    # Nó do Agente (Raciocínio com Gerenciador de Contexto/Tokens)
    def no_agente(state: AgentState) -> Dict[str, Any]:
        todas_mensagens = state["messages"]

        # Filtra apenas mensagens de conversação (não de sistema)
        mensagens_sem_system = [m for m in todas_mensagens if not isinstance(m, SystemMessage)]

        # Aplica a Janela Deslizante (Sliding Window): mantemos até as últimas 8 mensagens no contexto
        janela = mensagens_sem_system[-8:] if len(mensagens_sem_system) > 8 else mensagens_sem_system

        # Garantia de Integridade de Tool Calling:
        # Se a janela começar com uma ToolMessage órfã, removemos até encontrar uma HumanMessage ou AIMessage
        while janela and isinstance(janela[0], ToolMessage):
            janela = janela[1:]

        # Truncagem de Texto (Token Saver): Resume relatórios longos de turnos anteriores
        contexto_otimizado: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        
        for idx, msg in enumerate(janela):
            # Se for uma resposta em texto do assistente em um turno anterior com mais de 200 caracteres, trunca
            is_ultima_msg = (idx == len(janela) - 1)
            if isinstance(msg, AIMessage) and len(msg.content) > 200 and not msg.tool_calls and not is_ultima_msg:
                conteudo_resumido = msg.content[:200] + "... [resumo do histórico]"
                contexto_otimizado.append(AIMessage(content=conteudo_resumido))
            else:
                contexto_otimizado.append(msg)

        resposta_llm = llm_com_tools.invoke(contexto_otimizado)
        return {"messages": [resposta_llm]}

    # Instancia o Grafo de Estados
    builder = StateGraph(AgentState)

    # Adiciona os Nós (Nodes)
    builder.add_node("agent", no_agente)
    builder.add_node("tools", ToolNode(ferramentas_catalogadas))

    # Adiciona as Arestas (Edges)
    builder.add_edge(START, "agent")
    
    # Aresta Condicional: Se o LLM solicitou ferramenta, vai para 'tools', senão vai para END
    builder.add_conditional_edges("agent", tools_condition)
    
    # Após executar a ferramenta, retorna para o agente consolidar a resposta
    builder.add_edge("tools", "agent")

    return builder.compile()


# ==============================================================================
# 5. CAMADA DE VALIDAÇÃO DE RESPOSTA
# ==============================================================================
def validar_resposta_do_agente(resposta: str, eh_relatorio_completo: bool = False) -> Dict[str, Any]:
    """Inspeciona a resposta gerada pelo LangGraph para garantir qualidade e integridade."""
    erros = []

    if not resposta or len(resposta.strip()) < 10:
        erros.append("Resposta gerada está vazia ou excessivamente curta.")

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
# 6. CHAT INTERATIVO (CLI)
# ==============================================================================
if __name__ == "__main__":
    print("==================================================")
    print("  EcoAgent BR (LangGraph) - Chat Interativo Otimizado")
    print("  Estudo de Caso: Agente Baseado em Máquina de Estados (Grafo)")
    print("  Digite 'sair', 'exit' ou 'quit' para encerrar.")
    print("==================================================")

    # Compila o Grafo executável
    app = construir_grafo_ecoagent()

    # Histórico de estado mantido na sessão
    historico_mensagens: List[BaseMessage] = []

    while True:
        try:
            entrada_usuario = input("\nVocê: ").strip()

            if entrada_usuario.lower() in ["sair", "exit", "quit"]:
                print("Encerrando a sessão do EcoAgent BR (LangGraph). Até logo!")
                break

            if not entrada_usuario:
                continue

            # Adiciona a mensagem do usuário ao histórico
            historico_mensagens.append(HumanMessage(content=entrada_usuario))

            # Executa o Grafo passando o estado atual
            resultado = app.invoke({"messages": historico_mensagens})

            # Atualiza o histórico com o estado retornado pelo Grafo
            historico_mensagens = resultado["messages"]

            # Extrai o texto da última mensagem gerada pelo assistente
            resposta_final = historico_mensagens[-1].content

            # Validação pós-execução
            palavras_relatorio = ["investimento", "relatorio", "panorama", "melhor", "estrategia"]
            eh_relatorio_completo = any(p in entrada_usuario.lower() for p in palavras_relatorio)

            validacao = validar_resposta_do_agente(resposta_final, eh_relatorio_completo=eh_relatorio_completo)

            if not validacao["valido"]:
                print("\n[ALERTA DE VALIDAÇÃO]: Inconsistências encontradas:")
                for erro in validacao["erros"]:
                    print(f"  - ⚠️ {erro}")
                print("--------------------------------------------------")

            print("\nRESPOSTA FINAL DO AGENTE:")
            print(resposta_final)
            print("=" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nSessão encerrada pelo usuário.")
            break
        except Exception as erro:
            print(f"\n[ERRO CRÍTICO] Falha na execução do LangGraph: {erro}")