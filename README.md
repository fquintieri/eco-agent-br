# EcoAgent BR: Estudo Comparativo de Frameworks de Agentes de IA

Este repositório é uma demonstração pública e um estudo de caso prático focado na construção, avaliação e comparação de Agentes de IA em Python. O objetivo principal do projeto é resolver o mesmo problema de negócio utilizando 4 abordagens e frameworks distintos, analisando os trade-offs de controle, latência, consumo de tokens, resiliência e previsibilidade para ambientes de produção.

---

## Visão Geral da Arquitetura

O **EcoAgent BR** funciona como um assistente de inteligência macroeconômica brasileira. Ele consulta dados oficiais e atualizados em tempo real diretamente da API do Banco Central do Brasil (SGS - Sistema Gerenciador de Séries Temporais e PTAX):

* **Taxa SELIC Meta:** Indicador oficial de juros.
* **IPCA Mensal:** Índice de inflação mais recente.
* **Cotação do Dólar Comercial (PTAX):** Taxa de câmbio oficial de venda.

### Funcionamento do Agente
1. **Guardrail de Escopo:** O agente identifica se a pergunta é sobre economia brasileira. Solicitações fora do escopo (ex: clima, futebol, receitas ou mercado estrangeiro) são recusadas sem a execução desnecessária de ferramentas.
2. **Atendimento Adaptativo:**
   * **Saudações ou apresentações:** Resposta curta e direta em texto simples.
   * **Dúvidas pontuais de indicadores:** Chamada de ferramenta direcionada e resposta em parágrafo único.
   * **Análise de investimentos ou panorama:** Execução de todas as ferramentas e geração de relatório executivo em Markdown contendo tabela formatada e cálculo aproximado do juro real.

---

## Abordagens Implementadas

Implementamos a mesma solução sob 4 abordagens distintas para avaliar na prática o comportamento de cada uma:

* **Python Puro:** Construção do loop de raciocínio do zero para entender a mecânica sem abstrações.
* **Smolagents:** Leveza, sintaxe limpa e foco em execução ágil centrada em código.
* **CrewAI:** Agentes baseados em personas e papéis (*role-based squads*).
* **LangGraph:** Máquina de estados finita baseada em nós e arestas direcionadas.

---

## Principais Lições Aprendidas

### 1. O Custo Oculto da Abstração (*Token Bloat*)
* **O que vimos:** No **Python Puro**, enviávamos apenas a mensagem do usuário e o schema das 3 funções. No **CrewAI**, atingimos um erro de *Rate Limit (HTTP 429)* rapidamente durante os testes.
* **A Lição:** Frameworks de alto nível embutem uma quantidade gigantesca de *boilerplate* (instruções internas do framework, acompanhamento de tarefas, regras de formato e schemas adicionais) em todas as requisições. Isso infla o consumo de tokens de entrada em até 3x ou 4x. Em produção, abstração em excesso afeta diretamente os custos operacionais.

### 2. A Incompatibilidade dos Schemas de Ferramentas (*Tool Calling*)
* **O que vimos:** Ocorreu o erro `invalid JSON schema` no CrewAI ao integrar com a Groq, onde funções sem parâmetros enviavam a chave `"required"` sem a chave `"properties"`.
* **A Lição:** Provedores de API diferentes (OpenAI, Groq, Ollama, Anthropic) possuem validadores de JSON Schema com níveis de rigor distintos. Quando o framework gera o schema automaticamente sem flexibilidade, ele pode criar um contrato inválido para o seu provedor de LLM. O Python puro permite ajustar esse contrato manualmente.

### 3. A Regra do "Quem Sou Eu" vs. "O Que Devo Fazer"
* **O que vimos:** Quando regras de formato do relatório foram inseridas dentro do `backstory` no CrewAI, o modelo tentou antecipar a resposta e gerou tags como `<function=...>` como texto antes de rodar o código Python.
* **A Lição:** Em frameworks baseados em papéis, a **Persona** (`backstory`) serve para dar contexto de atuação. O **Procedimento** (`task`) deve ser estritamente sequencial: *1º Colete os dados; 2º Formate a saída*. Misturar persona com procedimento quebra o loop ReAct.

### 4. O Vício do "Over-Engineering" da LLM
* **O que vimos:** Ao perguntar *"qual a cotação do dólar?"*, o agente em suas primeiras versões executava todas as 3 ferramentas e montava um relatório completo com taxa SELIC e IPCA.
* **A Lição:** Se a intenção não for estritamente limitada, o modelo tentará entregar mais dados do que o solicitado. É necessário ensinar o agente a classificar a complexidade do pedido (Pergunta Direta vs. Relatório Executivo vs. Fora de Escopo) antes de acionar o catálogo de ferramentas.

### 5. Memória Não É Grátis (*Context Truncation*)
* **O que vimos:** No Smolagents, a tentativa de acessar `agent.logs` falhou, revelando que a estrutura interna correta era `agent.memory.steps`.
* **A Lição:** Deixar a memória acumular indefinidamente destrói a latência, estoura a janela de contexto e eleva custos. Inspecionar os objetos internos do framework para aplicar a truncagem de memória (*slicing* das últimas $N$ interações) é indispensável para criar conversas sustentáveis.

### 6. A "Ansiedade" da LLM (*Single-Turn Eagerness*) e Separação de Turnos
* **O que vimos:** Ao utilizar modelos open-weights (como a família Llama 3), exigir a formatação final de um relatório (tabelas em Markdown) no mesmo turno em que o modelo deve acionar uma ferramenta faz com que ele tenda a imprimir o nome da função como texto em vez de executá-la.
* **A Lição:** Para evitar falhas em *Tool Calling*, a melhor abordagem no prompt/task é dividir o fluxo em duas fases bem definidas: **Fase 1 (Coleta Silenciosa)** → **Fase 2 (Apresentação e Formatação)**.

### 7. Determinismo vs. Caixa-Preta em Produção
* **O que vimos:** Arquiteturas orientadas a tarefas e texto livre podem exigir ajustes constantes de prompt ao mudar de modelo.
* **A Lição:** Arquiteturas baseadas em Máquina de Estados (como o LangGraph) oferecem mais controle, previsibilidade e facilidade de depuração para sistemas que precisam ir para ambiente de produção.

---

## Guia de Instalação e Execução

### Pré-requisitos

* Python 3.10 ou superior
* Chave de API de um provedor LLM compatível (ex: Groq, OpenAI)

### 1. Clonar o Repositório

```bash
git clone https://github.com/SEU-USUARIO/eco-agent-br.git
cd eco-agent-br
```

### 2. Criar e Ativar o Ambiente Virtual

No Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

No Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar as Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurar as Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com base no arquivo `.env.example`:

```
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=sua_chave_de_api_aqui
LLM_MODEL=llama-3.3-70b-versatile
```

### 5. Executar as Abordagens

Python Puro:

```bash
python .\python_pure\main.py
```

Smolagents:

```bash
python .\smolagents\main.py
```

CrewAI:

```bash
python .\crewai\main.py
```

LangGraph:

```bash
python .\langgraph\main.py
```
---

## Estrutura do Projeto

```text
eco-agent-br/
├── shared/
│   ├── __init__.py
│   └── bcb_tools.py          # Integração com a API do BC
├── python_pure/
│   ├── __init__.py
│   └── main.py               # Implementação ReAct nativa em Python
├── smolagents/
│   ├── __init__.py
│   └── main.py               # Implementação utilizando Smolagents
├── crewai/
│   ├── __init__.py
│   └── main.py               # Implementação utilizando CrewAI
├── langgraph/
│   ├── __init__.py
│   └── main.py               # Implementação utilizando LangGraph
├── .env.example              # Modelo de variáveis de ambiente
├── .gitignore
├── requirements.txt          # Dependências do projeto
└── README.md                 # Documentação do repositório
```