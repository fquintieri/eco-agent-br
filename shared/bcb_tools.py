import json
import requests
from typing import Dict, Any, Callable

"""
Módulo de Ferramentas do Banco Central do Brasil (BCB)

Este módulo contém as funções responsáveis por realizar chamadas HTTP para o Sistema
Gerenciador de Séries Temporais (SGS) do Banco Central do Brasil.
"""

def get_economic_overview(*args, **kwargs) -> str:
    """Obtém SELIC, IPCA e Dólar de uma só vez e pré-calcula o Juro Real aproximado em Python."""
    try:
        # Executa as 3 consultas no Python
        selic_raw = json.loads(get_selic_rate())
        ipca_raw = json.loads(get_ipca_rate())
        usd_raw = json.loads(get_usd_exchange_rate())

        # Extrai e limpa os valores numéricos para cálculo
        def limpar_valor(val_str: str) -> float:
            texto = (
                val_str.replace("% ao ano", "")
                .replace("%", "")
                .replace(" BRL", "")
                .strip()
            )
            return float(texto.replace(",", "."))

        v_selic = limpar_valor(selic_raw["valor_atual"])
        v_ipca_mensal = limpar_valor(ipca_raw["valor_atual"])

        # Cálculo matemático determinístico em Python
        ipca_anualizado = v_ipca_mensal * 12
        juro_real_aprox = v_selic - ipca_anualizado

        # Estrutura unificada com cálculos já prontos
        payload = {
            "indicadores": {
                "selic": selic_raw,
                "ipca": ipca_raw,
                "dolar": usd_raw,
            },
            "calculos_pre_processados": {
                "ipca_anualizado_aprox": f"{ipca_anualizado:.2f}%",
                "juro_real_aprox": f"{juro_real_aprox:.2f}%",
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"erro": f"Falha ao gerar panorama econômico: {str(e)}"},
            ensure_ascii=False,
        )

def _consultar_sgs(codigo_serie: int, nome_indicador: str, sufixo_valor: str = "%") -> str:
    """
    Função auxiliar privada para realizar requisições genéricas ao SGS/BCB.
    Retorna uma string formatada em JSON estruturado para garantir fidelidade de dados.
    """
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados/ultimos/1?formato=json"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            payload = {
                "indicador": nome_indicador,
                "valor_atual": f"{data[0]['valor']}{sufixo_valor}",
                "data_referencia_oficial": data[0]['data']
            }
            return json.dumps(payload, ensure_ascii=False)
        
        return json.dumps({
            "erro": f"Status HTTP {response.status_code} ao buscar {nome_indicador}"
        }, ensure_ascii=False)

    except requests.exceptions.Timeout:
        return json.dumps({
            "erro": f"Timeout na conexão com o Banco Central ao buscar {nome_indicador}"
        }, ensure_ascii=False)
    except requests.exceptions.RequestException as e:
        return json.dumps({
            "erro": f"Falha na conexão com a API do Banco Central ao buscar {nome_indicador}: {str(e)}"
        }, ensure_ascii=False)


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

TOOLS_CATALOG: Dict[str, Callable[..., str]] = {
    "get_selic_rate": get_selic_rate,
    "get_ipca_rate": get_ipca_rate,
    "get_usd_exchange_rate": get_usd_exchange_rate,
    "get_economic_overview": get_economic_overview,  # <--- Nova Tool
}

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
            "description": "Obtém o índice de inflação oficial do Brasil (IPCA) acumulado do Banco Central.",
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
    },
    {
    "type": "function",
    "function": {
        "name": "get_economic_overview",
        "description": "Obtém um panorama completo com SELIC, IPCA, Dólar e o Juro Real pré-calculado. Use esta ferramenta SEMPRE que o usuário pedir um relatório, panorama, análise geral ou onde investir.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
}
]