from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#creating prompt template/input
prompt = PromptTemplate(
    template = 'generaaTE 5 intresting facts about {topic}',
    input_variables= ['topic']
)

#initialize model using huggingface

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task= "text-generation"
)


model = ChatHuggingFace(llm = llm)

#initialising the parser

parser = StrOutputParser()

# create a chain

chain = prompt | model | parser

#invoke the chain
result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()