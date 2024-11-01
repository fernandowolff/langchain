
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
import os
import pandas as pd
import json
from typing import List

class ExtratorDeUniversidade(BaseModel):
    universidade:str = Field("Nome da universidade em letras minúsculas.")

class DadosDeUniversidade(BaseTool):
    name = "DadosDeUniversidade"
    """Esta ferramenta extrai os dados de uma universidade. Passe para essa ferramenta como argumento o nome da universidade."""

    def _run(self, input: str) -> str:
        llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

        parser = JsonOutputParser(pydantic_object=ExtratorDeUniversidade)

        template = PromptTemplate(template="""
            Você deve analisar a entrada a seguir e extrair o nome da universidade informada em minúsculo.
            Entrada: 
            -----------
            {input}
            -----------
            Formato de saída:
            {formato_saida}
        """,
        input_variables=["input"],
        partial_variables={"formato_saida" : parser.get_format_instructions()})

        cadeia = template | llm | parser
        resposta = cadeia.invoke({"input": input})

        universidade = resposta['universidade'].lower().strip()
        dados = self.busca_dados_da_universidade(universidade)

        return json.dumps(dados)

    def busca_dados_da_universidade(universidade: str):
            dados = pd.read_csv("documentos/universidades.csv")
            dados['NOME_FACULDADE'] = dados["NOME_FACULDADE"].str.lower()
            dados_com_essa_universidade = dados[dados["NOME_FACULDADE"] == universidade]
    
            if dados_com_essa_universidade.empty:
                return {}
        
            return dados_com_essa_universidade.iloc[:1].to_dict()


class TodasUniversidades(BaseTool):
    name="TodasUniversidades"
    description="""Carrega os dados de todas as universidades. Não é necessário nenhum parâmetro de entrada."""

    def _run(self, input: str):
        universidades = self.busca_dados_das_universidades()

        return universidades

    def busca_dados_das_universidades():
            dados = pd.read_csv("documentos/universidades.csv")        
            return dados.to_dict()

