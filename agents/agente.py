
from langchain.agents import create_openai_tools_agent, create_react_agent
from langchain.agents import Tool
from langchain import hub
from langchain_openai import ChatOpenAI
import os
from estudante import DadosDeEstudante, PerfilAcademico
from universidade import DadosDeUniversidade, TodasUniversidades


class AgenteOpenAIFunctions:
    def __init__(self):
        
        llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

        dados_de_estudante = DadosDeEstudante()
        perfil_academico = PerfilAcademico()
        dados_da_universidade = DadosDeUniversidade()
        dados_todas_universidades = TodasUniversidades()
        self.tools = [
            Tool(name=dados_de_estudante.name, func = dados_de_estudante.run, description = dados_de_estudante.description, return_direct = False),
            Tool(name=perfil_academico.name, func = perfil_academico.run, description = perfil_academico.description),
            Tool(name=dados_da_universidade.name, func = dados_da_universidade.run, description = dados_da_universidade.description),
            Tool(name=dados_todas_universidades.name, func = dados_todas_universidades.run, description = dados_todas_universidades.description),
        ]

        # open ai
        #prompt = hub.pull("hwchase17/openai-functions-agent")
        #self.agente = create_openai_tools_agent(llm, self.tools, prompt)

        # react
        prompt = hub.pull("hwchase17/react")
        self.agente = create_react_agent(llm, self.tools, prompt)