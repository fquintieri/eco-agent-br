import os
import sys
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import inspect

# No Windows o console padrão usa cp1252 e quebra acentos/emojis; forçamos UTF-8.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

# Coloca a raiz do projeto no sys.path para que 'shared.bcb_tools' seja importável
# independentemente do diretório de onde o script for chamado.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importa os esquemas em formato JSON Schema (TOOLS_SCHEMA) e o mapa de funções executáveis (TOOLS_CATALOG).
from shared.bcb_tools import TOOLS_SCHEMA, TOOLS_CATALOG

# Carrega as variáveis de ambiente definidas no arquivo .env localizado na raiz do projeto.
load_dotenv()


def obter_cliente_openai() -> tuple[OpenAI, str]:
    """Instancia o cliente OpenAI a partir do .env e devolve (cliente, modelo).

    Usamos a SDK da OpenAI como protocolo comum: o mesmo código fala com Groq,
    Ollama, vLLM ou qualquer backend compatível bastando trocar base_url/api_key.
    O fallback aponta para um Ollama local para quem não tem chave de nuvem.
    """
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    model_name = os.getenv("LLM_MODEL", "qwen2.5:7b")

    # Sanitiza caso o valor no .env venha com o prefixo 'llm_model='
    if "llm_model=" in model_name.lower():
        model_name = model_name.split("=")[-1]

    print(f"[CONFIGURAÇÃO] Endpoint: {base_url}")
    print(f"[CONFIGURAÇÃO] Modelo Selecionado: {model_name}")

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model_name


# O System Prompt concentra todo o comportamento do agente: escopo, política de uso
# de ferramentas e formato de saída. Os quatro módulos deste estudo compartilham as
# mesmas quatro regras (guardrail de escopo, saudação, consulta pontual e relatório).
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
   - Responda em 1 parágrafo direto com o valor e a data de referência. NÃO monte tabelas.

4. RELATÓRIOS E PANORAMAS (ex: "Panorama econômico", "Onde investir?", "Como está a economia?"):
   - Acione TODAS as 3 ferramentas (`get_selic_rate`, `get_ipca_rate`, `get_usd_exchange_rate`) e só responda após coletar os três retornos.
   - Estruture a resposta em Markdown:
     * **Resumo Executivo**: um parágrafo sobre o cenário macroeconômico.
     * **Tabela de Indicadores**: Indicador, Valor Atual e Data de Referência.
     * **Análise de Impacto**: Juro Real aproximado (SELIC - IPCA) e leitura de Renda Fixa vs. Câmbio.

REGRA TRANSVERSAL: nunca anuncie que "vai consultar" — apenas execute as chamadas. Responda sempre em português do Brasil e use SOMENTE os valores retornados pelas ferramentas, sem inventar números.
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
    """Roda o loop ReAct (Reasoning + Acting) escrito na mão, com suporte a histórico e truncagem.

    O ciclo por passo é: enviar o histórico + os schemas das ferramentas -> ler a
    decisão do modelo -> se ele pediu ferramentas, executá-las localmente e devolver
    a observação ao histórico do turno -> repetir; se ele respondeu em texto, encerrar.
    """
    client, model_name = obter_cliente_openai()

    # Inicia a estrutura de mensagens enviadas na requisição
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Injeta o histórico recente com truncagem inteligente para economizar tokens
    if historico:
        for msg in historico[-4:]:  # Mantém até os últimos 2 turnos da conversa
            conteudo = msg["content"]
            if len(conteudo) > 200:
                conteudo = conteudo[:200] + "... [resumo do histórico]"
            messages.append({"role": msg["role"], "content": conteudo})

    # Adiciona a pergunta atual do usuário
    messages.append({"role": "user", "content": pergunta_usuario})

    print("\n--- INICIO DO AGENT LOOP ---")
    print(f"Pergunta do Usuario: '{pergunta_usuario}'\n")

    # ==========================================================================
    # LAÇO PRINCIPAL DE EXECUÇÃO (AGENT LOOP / REACT CYCLE)
    # ==========================================================================
    for passo in range(1, max_passos + 1):
        print(
            f"[PASSO {passo}/{max_passos}] Enviando contexto com {len(messages)} mensagem(ns) para a LLM..."
        )

        try:
            # 1. ETAPA DE RACIOCÍNIO (REASONING):
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOLS_SCHEMA,  # Catálogo de schemas JSON enviado à LLM
                tool_choice="auto",  # A LLM decide autonomamente se usa ou não ferramentas
                temperature=0.0,  # Temperatura 0.0 para máximo determinismo e estabilidade
            )
        except Exception as erro:
            mensagem_erro = f"Erro na comunicação com a API da LLM ({model_name}): {str(erro)}"
            print(f"[ERRO DE CONEXÃO] {mensagem_erro}")
            return mensagem_erro

        # Extrai a mensagem de resposta gerada pela LLM nesta iteração
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Adiciona a resposta do assistente ao histórico do turno
        messages.append(response_message)

        # ======================================================================
        # CONDIÇÃO DE RAMIFICAÇÃO 1: A LLM SOLICITOU EXECUÇÃO DE FERRAMENTAS (AÇÃO)
        # ======================================================================
        if tool_calls:
            print(
                f"[RACIOCÍNIO DA LLM] A LLM identificou a necessidade de executar {len(tool_calls)} ferramenta(s)."
            )

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                tool_call_id = tool_call.id
                raw_arguments = tool_call.function.arguments

                # Parsing defensivo de argumentos JSON
                if raw_arguments:
                    try:
                        function_args = json.loads(raw_arguments)
                    except Exception:
                        function_args = {}
                else:
                    function_args = {}

                if not isinstance(function_args, dict):
                    function_args = {}

                print(
                    f"[AÇÃO DO HARNESS] Solicitando execução da função local: '{function_name}' com parâmetros: {function_args}"
                )

                # Executa a função localmente se ela existir no catálogo
                if function_name in TOOLS_CATALOG:
                    funcao_python = TOOLS_CATALOG[function_name]

                    # Inspeciona a assinatura real da função Python
                    sig = inspect.signature(funcao_python)

                    # Se a função não espera nenhum parâmetro (caso da get_selic_rate)
                    if len(sig.parameters) == 0:
                        resultado_ferramenta = funcao_python()
                    else:
                        # Se a função espera parâmetros, filtra apenas os que ela aceita de fato
                        args_validos = {
                            k: v for k, v in function_args.items() if k in sig.parameters
                        }
                        resultado_ferramenta = funcao_python(**args_validos)
                else:
                    resultado_ferramenta = (
                        f"Erro: A ferramenta '{function_name}' não existe no catálogo do sistema."
                    )

                # 2. ETAPA DE FEEDBACK / OBSERVAÇÃO (OBSERVATION):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": str(resultado_ferramenta),
                })

        # ======================================================================
        # CONDIÇÃO DE RAMIFICAÇÃO 2: A LLM GEROU UMA RESPOSTA EM TEXTO (CONCLUSÃO)
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
    # CONDIÇÃO DE PARADA POR EXCEDER O LIMITE DE ITERAÇÕES (MAX STEPS)
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

    # Armazena a memória de sessão acumulada do chat no terminal
    historico_sessao: List[Dict[str, str]] = []

    while True:
        try:
            entrada_usuario = input("\nVocê: ").strip()

            if entrada_usuario.lower() in ["sair", "exit", "quit"]:
                print("Encerrando a sessão do EcoAgent BR. Até logo!")
                break

            if not entrada_usuario:
                continue

            # Passa a entrada atual juntamente com o histórico mantido no Python puro
            resposta_final = executar_agente(entrada_usuario, historico_sessao)

            # Atualiza o histórico acumulado para as próximas rodadas
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