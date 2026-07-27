import os
import sys
import json
import re
import inspect
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI


# Adiciona a raiz do projeto ao sys.path para importação de 'shared.bcb_tools'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importa os esquemas JSON Schema e o catálogo executável de funções Python
from shared.bcb_tools import TOOLS_SCHEMA, TOOLS_CATALOG

# Carrega variáveis de ambiente (.env)
load_dotenv()


def obter_cliente_openai() -> tuple[OpenAI, str]:
    """Instancia o cliente OpenAI a partir do .env e devolve (cliente, modelo).

    A SDK da OpenAI atua como protocolo comum para Groq, Ollama, vLLM ou OpenAI.
    """
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    model_name = os.getenv("LLM_MODEL", "qwen2.5:7b")

    # Sanitização de string caso venha com o prefixo 'llm_model='
    if "llm_model=" in model_name.lower():
        model_name = model_name.split("=")[-1]

    print(f"[CONFIGURAÇÃO] Endpoint: {base_url}")
    print(f"[CONFIGURAÇÃO] Modelo Selecionado: {model_name}")

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model_name


# ==============================================================================
# PROMPT DO SISTEMA (GUARDRAILS & REGRAS DE COMPORTAMENTO)
# ==============================================================================
SYSTEM_PROMPT = """
Você é o **EcoAgent BR**, um analista macroeconômico sênior especializado EXCLUSIVAMENTE em indicadores do Banco Central do Brasil e finanças.

1. TRAVA DE ESCOPO (GUARDRAIL ABSOLUTO):
   - Seu domínio é ESTRITAMENTE: economia brasileira, taxa SELIC, IPCA, Dólar, mercado financeiro e investimentos.
   - Para assuntos fora desse domínio (esportes, clima, receitas, curiosidades), NÃO chame ferramentas e recuse com:
     "Como sou o EcoAgent BR, meu foco é exclusivamente a análise econômica do Brasil e indicadores do Banco Central. Como posso te ajudar com a taxa SELIC, IPCA, Dólar ou estratégias de investimentos hoje?"

2. SAUDAÇÕES E IDENTIDADE:
   - Para 'oi', 'olá' ou 'quem é você?': NÃO chame ferramentas. Responda em 1 frase curta, apresentando-se e citando que consulta SELIC, IPCA e Dólar.

3. CONSULTA PONTUAL DE UM INDICADOR (ex: "Qual a SELIC?", "Cotação do Dólar"):
   - Acione IMEDIATAMENTE apenas a ferramenta correspondente.

4. RELATÓRIOS E PANORAMAS (ex: "Panorama econômico", "Onde investir?", "Gere um relatório"):
   - Acione TODAS as 3 ferramentas (`get_selic_rate`, `get_ipca_rate`, `get_usd_exchange_rate`) e só responda após coletar os três retornos.
   - Estruture a resposta estritamente em Markdown:
     * **Resumo Executivo**: um parágrafo sobre o cenário macroeconômico.
     * **Tabela de Indicadores**: Indicador, Valor Atual e Data de Referência.
     * **Análise de Impacto**:
       - Nota: O IPCA é MENSUAL e a SELIC é ANUAL. Para o Juro Real aproximado, considere a inflação anualizada (IPCA mensal x 12) antes da subtração.
       - Leitura prática para Renda Fixa vs. Câmbio.
    4. RELATÓRIOS E PANORAMAS (ex: "Panorama econômico", "Onde investir?", "Gere um relatório"):
   - Acione APENAS a ferramenta `get_economic_overview`.
   - Utilize rigorosamente os dados da chave `calculos_pre_processados` para citar a inflação anualizada e o juro real. NÃO refaça contas matemáticas.

5. REGRA DE EXECUÇÃO:
   - Assim que receber a observação das ferramentas (mensagens de 'tool'), você DEVE sintetizar a resposta final no turno seguinte.
   - É PROIBIDO chamar a mesma ferramenta consecutivamente na mesma rodada.

6. REGRA DE FIDELIDADE CRÍTICA AOS DADOS (GROUNDING ABSOLUTO):
   - COPIE E COLE RIGOROSAMENTE os valores numéricos e as datas de referência exatos fornecidos pelo JSON da ferramenta.
   - É ESTRITAMENTE PROIBIDO alterar datas ou utilizar números de sua memória de treinamento.

REGRA TRANSVERSAL: nunca anuncie que "vai consultar" — apenas execute as chamadas. Responda sempre em português do Brasil.
"""


def contem_degeneracao(texto: str) -> bool:
    """Verifica se o texto retornado contém caracteres fora do padrão PT-BR/EN.

    Mecanismo de defesa contra o colapso de amostragem (Degradação de Tokens).
    """
    if not texto:
        return False
    padrao_estranho = re.compile(
        r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uffef\u4e00-\u9faf]"
    )
    return bool(padrao_estranho.search(texto))


def executar_agente(
    pergunta_usuario: str,
    historico: List[Dict[str, str]] = None,
    max_passos: int = 5,
) -> str:
    """Roda o ciclo ReAct (Reasoning + Acting) em Python puro com suporte a:

    - Inspecção dinâmica de parâmetros (evita TypeError por alucinação).
    - Pass-Through determinístico para consultas pontuais (100% de precisão de dados).
    - Truncagem de histórico para economia de tokens.
    """
    client, model_name = obter_cliente_openai()

    # Inicia a estrutura de mensagens enviadas na requisição
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Injeta o histórico recente com truncagem inteligente
    if historico:
        for msg in historico[-4:]:  # Mantém os últimos 2 turnos da conversa
            conteudo = msg["content"]
            if len(conteudo) > 200:
                conteudo = conteudo[:200] + "... [resumo do histórico]"
            messages.append({"role": msg["role"], "content": conteudo})

    # Adiciona a pergunta atual do usuário
    messages.append({"role": "user", "content": pergunta_usuario})

    print("\n--- INICIO DO AGENT LOOP ---")
    print(f"Pergunta do Usuario: '{pergunta_usuario}'\n")

    # ==========================================================================
    # LAÇO PRINCIPAL DE EXECUÇÃO (REACT CYCLE)
    # ==========================================================================
    for passo in range(1, max_passos + 1):
        print(
            f"[PASSO {passo}/{max_passos}] Enviando contexto com {len(messages)} mensagem(ns) para a LLM..."
        )

        try:
            # 1. ETAPA DE RACIOCÍNIO (REASONING)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.0,  # Máxima estabilidade
            )
        except Exception as erro:
            mensagem_erro = f"Erro na comunicação com a API da LLM ({model_name}): {str(erro)}"
            print(f"[ERRO DE CONEXÃO] {mensagem_erro}")
            return mensagem_erro

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Registra a intenção do assistente no histórico
        messages.append(response_message)

        # ======================================================================
        # CONDIÇÃO 1: A LLM SOLICITOU EXECUÇÃO DE FERRAMENTAS (AÇÃO)
        # ======================================================================
        if tool_calls:
            print(
                f"[RACIOCÍNIO DA LLM] A LLM identificou a necessidade de executar {len(tool_calls)} ferramenta(s)."
            )

            # Flag para identificar se é uma consulta pontual de 1 único indicador
            eh_consulta_pontual = len(tool_calls) == 1

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                tool_call_id = tool_call.id
                raw_arguments = tool_call.function.arguments

                # Parsing defensivo de argumentos JSON
                try:
                    function_args = json.loads(raw_arguments) if raw_arguments else {}
                except Exception:
                    function_args = {}

                if not isinstance(function_args, dict):
                    function_args = {}

                print(
                    f"[AÇÃO DO HARNESS] Solicitando execução da função local: '{function_name}' com parâmetros: {function_args}"
                )

                # Execução segura via inspecção de assinatura
                if function_name in TOOLS_CATALOG:
                    funcao_python = TOOLS_CATALOG[function_name]
                    sig = inspect.signature(funcao_python)

                    try:
                        # Se a função não aceita parâmetros, ignora o que a LLM enviou
                        if len(sig.parameters) == 0:
                            resultado_ferramenta = funcao_python()
                        else:
                            # Filtra apenas os argumentos válidos da assinatura Python
                            args_validos = {
                                k: v for k, v in function_args.items() if k in sig.parameters
                            }
                            resultado_ferramenta = funcao_python(**args_validos)
                    except Exception as err_exec:
                        resultado_ferramenta = json.dumps(
                            {"erro": f"Erro de execução na ferramenta '{function_name}': {str(err_exec)}"},
                            ensure_ascii=False
                        )
                else:
                    resultado_ferramenta = json.dumps(
                        {"erro": f"A ferramenta '{function_name}' não existe no catálogo."},
                        ensure_ascii=False
                    )

                print(
                    f"[OBSERVAÇÃO DA FERRAMENTA] Retorno obtido: {resultado_ferramenta}"
                )

                # ==============================================================
                # INTERCEPÇÃO DETERMINÍSTICA (PASS-THROUGH PARA CONSULTAS PONTUAIS)
                # ==============================================================
                # Se for consulta de 1 indicador, formatamos direto no Python e finalizamos.
                # Isso elimina 100% o risco de a LLM alucinar ou reescrever a data/valor.
                if eh_consulta_pontual:
                    try:
                        dados = json.loads(resultado_ferramenta)
                        if "valor_atual" in dados and "data_referencia_oficial" in dados:
                            print(
                                "[DETERMINÍSTICO] Resposta pontual gerada diretamente pelo código (Pass-Through)."
                            )
                            print("--- FIM DO AGENT LOOP ---\n")
                            return (
                                f"A **{dados['indicador']}** é de **{dados['valor_atual']}** "
                                f"(Data de referência: {dados['data_referencia_oficial']})."
                            )
                    except Exception:
                        # Se o retorno não for um JSON padrão, prossegue no fluxo ReAct
                        pass

                # Alimenta a observação no histórico para síntese de relatórios complexos
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": str(resultado_ferramenta),
                })

        # ======================================================================
        # CONDIÇÃO 2: A LLM FINALIZOU A RESPOSTA EM TEXTO (SÍNTESE DO RELATÓRIO)
        # ======================================================================
        else:
            print(
                "[CONCLUÍDO] A LLM finalizou o raciocínio sem solicitar novas ferramentas."
            )
            print("--- FIM DO AGENT LOOP ---\n")

            conteudo_resposta = response_message.content

            # Tratamento para retorno nulo ou vazio
            if not conteudo_resposta or not conteudo_resposta.strip():
                conteudo_resposta = (
                    "Olá! Eu sou o **EcoAgent BR**, seu assistente especializado em indicadores econômicos do Banco Central.\n\n"
                    "Posso te ajudar a consultar em tempo real:\n"
                    "* **Taxa SELIC** (Taxa básica de juros)\n"
                    "* **IPCA** (Índice oficial de inflação)\n"
                    "* **Cotação do Dólar** (Taxa Ptax / Comercial)\n\n"
                    "Qual indicador você gostaria de verificar agora?"
                )

            # Filtro de sanitização contra degradação de tokens
            if contem_degeneracao(conteudo_resposta):
                print(
                    "[AVISO DE SEGURANÇA] O modelo gerou uma resposta corrompida/degenerada."
                )
                return "Ocorreu uma falha na geração da resposta pelo modelo. Por favor, refaça a pergunta de forma mais direta."

            return conteudo_resposta

    # ==========================================================================
    # CONDIÇÃO DE PARADA POR EXCEDER MAX STEPS
    # ==========================================================================
    print(
        "[AVISO] O agente atingiu o limite máximo de passos sem concluir a tarefa."
    )
    return "Não foi possível concluir a análise dentro do limite máximo de iterações estabelecido."


# ==============================================================================
# PONTO DE ENTRADA DO SCRIPT (CHAT INTERATIVO VIA TERMINAL / CLI)
# ==============================================================================
if __name__ == "__main__":
    print("==================================================")
    print("  EcoAgent BR (Python Puro) - Chat Interativo")
    print("  Estudo de Caso: Agente Autônomo ReAct sem Frameworks")
    print("  Digite 'sair', 'exit' ou 'quit' para encerrar.")
    print("==================================================")

    historico_sessao: List[Dict[str, str]] = []

    while True:
        try:
            entrada_usuario = input("\nVocê: ").strip()

            if entrada_usuario.lower() in ["sair", "exit", "quit"]:
                print("Encerrando a sessão do EcoAgent BR. Até logo!")
                break

            if not entrada_usuario:
                continue

            # Passa a entrada atual e o histórico mantido no Python
            resposta_final = executar_agente(entrada_usuario, historico_sessao)

            # Atualiza o histórico para as próximas rodadas
            historico_sessao.append(
                {"role": "user", "content": entrada_usuario}
            )
            historico_sessao.append(
                {"role": "assistant", "content": resposta_final}
            )

            print("\nRESPOSTA FINAL DO AGENTE:")
            print(resposta_final)
            print("=" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nSessão encerrada pelo usuário. Até logo!")
            break
        except Exception as erro_geral:
            print(
                f"\n[ERRO CRÍTICO] Ocorreu uma falha inesperada durante a execução: {erro_geral}"
            )