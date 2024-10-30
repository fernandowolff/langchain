from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class ExtratorDeEstudante(BaseModel):
    estudante:str = Field("Nome do estudante informado, sempre em letras minúsculas. Exemplos: joão, carla, ana.")

class DadosDeEstudante(BaseTool):
    name = "DadosDeEstudante"
    description = """Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico."""

    def _run(self, input: str) -> str:
        llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)

        template = PromptTemplate(template="""
            Você deve analisar a entrada e extrair o nome da pessoa que ela contém.
            Entrada: 
            {input}
            Formato de saída:
            {formato_saida}
        """,
        input_variables=["input"],
        partial_variables={"formato_saida" : parser.get_format_instructions()})

        cadeia = template | llm | parser
        resposta = cadeia.invoke({"input": input})

        return resposta['estudante']

pergunta = "Quais os dados da Ana?"

DadosDeEstudante().run(pergunta)

