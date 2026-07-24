import requests
from typing import Dict, Any, Callable

"""
Módulo de Ferramentas do Banco Central do Brasil (BCB)

Este módulo contém as funções responsáveis por realizar chamadas HTTP para o Sistema
Gerenciador de Séries Temporais (SGS) do Banco Central do Brasil.
"""

def _consultar_sgs(codigo_serie: int, nome_indicador: str, sufixo_valor: str = "%") -> str:
    """
    Função auxiliar privada para realizar requisições genéricas ao SGS/BCB.
    Evita duplicação de código HTTP e tratamento de exceções.
    """
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados/ultimos/1?formato=json"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            valor = data[0]['valor']
            data_registro = data[0]['data']
            return f"{nome_indicador}: {valor}{sufixo_valor} (Data de referência: {data_registro})."
        
        return f"Erro ao consultar {nome_indicador}. Código de status HTTP: {response.status_code}"

    except requests.exceptions.Timeout:
        return f"Erro: A conexão com o servidor do Banco Central esgotou o tempo limite (timeout) ao buscar {nome_indicador}."
    except requests.exceptions.RequestException as e:
        return f"Falha na conexão com a API do Banco Central ao buscar {nome_indicador}: {str(e)}"


# ==============================================================================
# FUNÇÕES PÚBLICAS DAS FERRAMENTAS
# ==============================================================================

def get_selic_rate() -> str:
    """Obtém a taxa SELIC meta atualizada do Banco Central do Brasil."""
    return _consultar_sgs(codigo_serie=432, nome_indicador="Taxa SELIC meta atual", sufixo_valor="% ao ano")


def get_ipca_rate() -> str:
    """Obtém o índice de inflação oficial do Brasil (IPCA) mensal mais recente do Banco Central."""
    return _consultar_sgs(codigo_serie=433, nome_indicador="IPCA mensal mais recente", sufixo_valor="%")


def get_usd_exchange_rate() -> str:
    """Obtém a cotação oficial do Dólar comercial (PTAX venda) em Reais (BRL) do Banco Central."""
    return _consultar_sgs(codigo_serie=10813, nome_indicador="Cotação do Dólar Comercial (PTAX venda)", sufixo_valor=" BRL")


# ==============================================================================
# MAPEAMENTO E SCHEMAS PARA FUNCTION CALLING
# ==============================================================================

# Catálogo dinâmico de funções (Chaves alinhadas com o Schema do Agent)
TOOLS_CATALOG: Dict[str, Callable[[], str]] = {
    "get_selic_rate": get_selic_rate,
    "get_ipca_rate": get_ipca_rate,
    "get_usd_exchange_rate": get_usd_exchange_rate
}

# JSON Schema padrão OpenAI
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_selic_rate",
            "description": "Obtém a taxa SELIC atualizada do Banco Central do Brasil.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ipca_rate",
            "description": "Obtém o índice de inflação oficial do Brasil (IPCA) acumulado do Banco Central. Use esta ferramenta sempre que o usuário perguntar por 'inflação', 'IPCA' ou 'índice de preços'.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_usd_exchange_rate",
            "description": "Obtém a cotação atual do Dólar Ptax/Comercial em Reais (BRL) do Banco Central.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]