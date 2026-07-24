import os
import sys
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# No Windows o console padrao usa cp1252 e quebra acentos/emojis; forcamos UTF-8.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

# Coloca a raiz do projeto no sys.path para que 'shared.bcb_tools' seja importavel
# independentemente do diretorio de onde o script for chamado.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importa os esquemas em formato JSON Schema (TOOLS_SCHEMA) e o mapa de funcoes executaveis (TOOLS_CATALOG).
from shared.bcb_tools import TOOLS_SCHEMA, TOOLS_CATALOG

# Carrega as variaveis de ambiente definidas no arquivo .env localizado na raiz do projeto.
load_dotenv()


def obter_cliente_openai() -> tuple[OpenAI, str]:
    """Instancia o cliente OpenAI a partir do .env e devolve (cliente, modelo).

    Usamos a SDK da OpenAI como protocolo comum: o mesmo codigo fala com Groq,
    Ollama, vLLM ou qualquer backend compativel bastando trocar base_url/api_key.
    O fallback aponta para um Ollama local para quem nao tem chave de nuvem.
    """
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    model_name = os.getenv("LLM_MODEL", "qwen2.5:7b")

    print(f"[CONFIGURACAO] Endpoint: {base_url}")
    print(f"[CONFIGURACAO] Modelo Selecionado: {model_name}")

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model_name


# O System Prompt concentra todo o comportamento do agente: escopo, política de uso
# de ferramentas e formato de saída. Os quatro módulos deste estudo compartilham as
# mesmas quatro regras (guardrail de escopo, saudação, consulta pontual e relatório),
# de modo que a comparação isole o framework — não o prompt.
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
    """
    Verifica se o texto retornado contem caracteres fora do padrao PT-BR/EN.
    Mecanismo de defesa contra o colapso de amostragem (Degradaçao de Tokens).
    """
    if not texto:
        return False
    # Detecta caracteres asiaticos (CJK), cirilicos ou simbolos desconexos
    padrao_estranho = re.compile(r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uffef\u4e00-\u9faf]')
    return bool(padrao_estranho.search(texto))


def executar_agente(pergunta_usuario: str, max_passos: int = 5) -> str:
    """Roda o loop ReAct (Reasoning + Acting) escrito na mao, sem framework.

    O ciclo por passo e: enviar o historico + os schemas das ferramentas -> ler a
    decisao do modelo -> se ele pediu ferramentas, executa-las localmente e devolver
    a observacao ao historico -> repetir; se ele respondeu em texto, encerrar.
    'max_passos' e o freio contra loops infinitos.
    """
    client, model_name = obter_cliente_openai()

    # A LLM e stateless: nao guarda nada entre requisicoes. Toda a memoria de curto
    # prazo vive nesta lista, reenviada por inteiro a cada iteracao do loop.
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pergunta_usuario}
    ]

    print("\n--- INICIO DO AGENT LOOP ---")
    print(f"Pergunta do Usuario: '{pergunta_usuario}'\n")

    # ==========================================================================
    # LAÇO PRINCIPAL DE EXECUCAO (AGENT LOOP / REACT CYCLE)
    # ==========================================================================
    for passo in range(1, max_passos + 1):
        print(f"[PASSO {passo}/{max_passos}] Enviando contexto com {len(messages)} mensagem(ns) para a LLM...")

        try:
            # 1. ETAPA DE RACIOCINIO (REASONING):
            # Envia todo o historico de mensagens e o esquema de ferramentas (TOOLS_SCHEMA).
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOLS_SCHEMA,        # Catalogo de schemas JSON enviado a LLM
                tool_choice="auto",        # A LLM decide autonomamente se usa ou nao ferramentas
                temperature=0.0            # Temperatura 0.0 para maximo determinismo e estabilidade
            )
        except Exception as erro:
            mensagem_erro = f"Erro na comunicacao com a API da LLM ({model_name}): {str(erro)}"
            print(f"[ERRO DE CONEXAO] {mensagem_erro}")
            return mensagem_erro

        # Extrai a mensagem de resposta gerada pela LLM nesta iteracao
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Adiciona a resposta do assistente (seja texto ou intencao de chamar ferramenta) ao historico.
        # Isso e obrigatorio para manter a integridade do protocolo da API da OpenAI.
        messages.append(response_message)

        # ======================================================================
        # CONDICAO DE RAMIFICACAO 1: A LLM SOLICITOU EXECUCAO DE FERRAMENTAS (ACAO)
        # ======================================================================
        if tool_calls:
            print(f"[RACIOCINIO DA LLM] A LLM identificou a necessidade de executar {len(tool_calls)} ferramenta(s).")

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                tool_call_id = tool_call.id
                raw_arguments = tool_call.function.arguments

                # --------------------------------------------------------------
                # PARSING DEFENSIVO DE ARGUMENTOS (MULTICLOUD & LOCAL SAFETY)
                # Dicionarios nulos (None) ou strings invalidas sao convertidos para {}
                # para evitar erros de desempacotamento (**args).
                # --------------------------------------------------------------
                if raw_arguments:
                    try:
                        function_args = json.loads(raw_arguments)
                    except Exception:
                        function_args = {}
                else:
                    function_args = {}

                # Garantia absoluta: Se o JSON deserializado nao for um dicionario Python, forca {}
                if not isinstance(function_args, dict):
                    function_args = {}

                print(f"[ACAO DO HARNESS] Solicitando execucao da funcao local: '{function_name}' com parametros: {function_args}")

                # Verifica se a funcao solicitada existe no nosso catalogo de funcoes registradas
                if function_name in TOOLS_CATALOG:
                    # Recupera a referencia da funcao Python e a executa passando os argumentos
                    funcao_python = TOOLS_CATALOG[function_name]
                    resultado_ferramenta = funcao_python(**function_args)
                else:
                    resultado_ferramenta = f"Erro: A ferramenta '{function_name}' nao existe no catalogo do sistema."

                print(f"[OBSERVACAO DA FERRAMENTA] Retorno obtido da API externa: {resultado_ferramenta}")

                # 2. ETAPA DE FEEDBACK / OBSERVACAO (OBSERVATION):
                # O resultado retornado e empacotado em uma nova mensagem com 'role': 'tool'.
                # E crucial incluir o 'tool_call_id' para sincronizacao do historico.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": str(resultado_ferramenta)
                })

            # O laço continuara para o proximo passo, enviando o resultado da API de volta a LLM.

        # ======================================================================
        # CONDICAO DE RAMIFICACAO 2: A LLM GEROU UMA RESPOSTA EM TEXTO (CONCLUSAO)
        # ======================================================================
        else:
            print("[CONCLUIDO] A LLM finalizou o raciocinio sem solicitar novas ferramentas.")
            print("--- FIM DO AGENT LOOP ---\n")

            conteudo_resposta = response_message.content

            # TRATAMENTO DE RETORNO NULO / VAZIO (Fallback alinhado a Skill):
            if not conteudo_resposta or not conteudo_resposta.strip():
                conteudo_resposta = (
                    "Ola! Eu sou o **EcoAgent BR**, seu assistente especializado em indicadores economicos do Banco Central.\n\n"
                    "Posso te ajudar a consultar em tempo real:\n"
                    "* **Taxa SELIC** (Taxa basica de juros)\n"
                    "* **IPCA** (Indice oficial de inflaçao)\n"
                    "* **Cotaçao do Dolar** (Taxa Ptax / Comercial)\n\n"
                    "Qual indicador voce gostaria de verificar agora?"
                )

            # FILTRO DE SANITIZAÇÃO: Impede a exibicao de respostas corrompidas por degradaçao de tokens
            if contem_degeneracao(conteudo_resposta):
                print("[AVISO DE SEGURANÇA] O modelo gerou uma resposta corrompida/degenerada.")
                return "Ocorreu uma falha na geracao da resposta pelo modelo. Por favor, refaça a pergunta de forma mais direta."

            return conteudo_resposta

    # ==========================================================================
    # CONDICAO DE PARDA POR EXCEDER O LIMITE DE ITERACOES (MAX STEPS)
    # ==========================================================================
    print("[AVISO] O agente atingiu o limite maximo de passos sem concluir a tarefa.")
    return "Nao foi possivel concluir a analise dentro do limite maximo de iteracoes estabelecido."


# ==============================================================================
# PONTO DE ENTRADA DO SCRIPT (CHAT INTERATIVO VIA TERMINAL / CLI)
# ==============================================================================
if __name__ == "__main__":
    print("==================================================")
    print("  EcoAgent BR (Python Puro) - Chat Interativo")
    print("  Estudo de Caso: Agente Autonomo ReAct sem Frameworks")
    print("  Digite 'sair', 'exit' ou 'quit' para encerrar.")
    print("==================================================")

    # Laço continuo para manter a sessao do terminal ativa
    while True:
        try:
            # Captura a entrada do usuario no terminal
            entrada_usuario = input("\nVoce: ").strip()

            # Condicao de saida do chat
            if entrada_usuario.lower() in ["sair", "exit", "quit"]:
                print("Encerrando a sessao do EcoAgent BR. Ate logo!")
                break

            # Ignora entradas vazias (pressionar Enter sem digitar nada)
            if not entrada_usuario:
                continue

            # Executa a rotina do agente para processar a pergunta
            resposta_final = executar_agente(entrada_usuario)

            # Exibe o resultado consolidado no console
            print("\nRESPOSTA FINAL DO AGENTE:")
            print(resposta_final)
            print("=" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nSessao encerrada pelo usuario. Ate logo!")
            break
        except Exception as erro_geral:
            print(f"\n[ERRO CRITICO] Ocorreu uma falha inesperada durante a execucao: {erro_geral}")