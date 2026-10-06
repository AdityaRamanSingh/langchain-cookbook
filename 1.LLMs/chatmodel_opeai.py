from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

# temperature is parameter that controlls the randomness of a language model output, it affects how creative or determinstic responses are.
chat_model = ChatOpenAI(model="gpt-4", temperature="")

result = chat_model.invoke("question")

print(result)