
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
from agente import AgenteOpenAIFunctions

load_dotenv()

agente = AgenteOpenAIFunctions()
executor = AgentExecutor(agente = agente.agente, tools = agente.tools, verbose=True)

pergunta = "Quais os dados da Ana?"
pergunta = "Quais os dados da Bianca?"
pergunta = "Quais os dados da Ana e da Bianca?"
pergunta = "Crie um perfil acadêmico para a Ana!"
pergunta = "Compare o perfil acadêmico da Ana com o da Bianca!"
pergunta = "Tenho sentido Ana desanimada com cursos de matemática. Seria uma boa parear ela com a Bianca?"
pergunta = "Tenho sentido Ana desanimada com cursos de matemática. Seria uma boa parear ela com o Marcos?"
pergunta = "Quais os dados da USP?"
pergunta = "Dentre USP e UFRJ, qual você recomenda para a acadêmica Ana?"
pergunta = "Dentre uni camp e USP, qual você recomenda para a Ana?"
pergunta = "Quais as faculdades com melhores chances para a Ana entrar?"
pergunta = "Dentre todas as faculdades existentes, quais Ana possui maiores chances de entrar?"
pergunta = "Além das faculdades favoritas da Ana, existem outras faculdades. Considere elas também. Quais Ana possui mais chance de entrar?"
resposta = executor.invoke({"input": pergunta})

print(resposta)

