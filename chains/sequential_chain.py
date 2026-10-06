from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


#prompt 1
prompt1 = PromptTemplate(
    template = 'generate a etailed report on {topic}',
    input_variables= ['topic']
)

#prompt 2
prompt2 = PromptTemplate(
    template = 'generate a 5 pointer summary on following \n {text}',
    input_variables= ['text']
)

#initialize model using huggingface
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task= "text-generation"
)


model = ChatHuggingFace(llm = llm)


#initialising the parser
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic' : 'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()