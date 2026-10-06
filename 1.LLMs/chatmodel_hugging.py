from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id="openai/gpt-oss-120b",
    task= "text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Write a python code to print a right trianlge using *")

print(result.content)