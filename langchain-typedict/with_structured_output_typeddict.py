from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task= "text-generation"
)



model = ChatHuggingFace(llm = llm)


# schema
class Review(TypedDict):
    key_themes : Annotated [list[str],"Write down all key themes in the review in a list"]
    summary: Annotated[str,"A brief summary of review"]
    sentiment: Annotated[str,"Return sentiment of review either positivem negative or neutral"]
    pros : Annotated[Optional[list[str]], "write all the pros in the list"]
    cons : Annotated[Optional[list[str]], "write all the cons in the list"]
    name : Annotated[Optional[str], "Write the name of reviewer"]

structured_model = model.with_structured_output(Review)


result = structured_model.invoke(
    """"
    I have been using this laptop for a few months, and overall I am
    quite satisfied with it. The performance is smooth for everyday tasks
    like browsing, coding, watching videos, and working with documents.
    The SSD makes it boot quickly, and the keyboard is comfortable for
    long typing sessions. I also like the lightweight design because it
    is easy to carry.

    However, the battery life is only average, especially when running
    heavy applications. The display is good indoors but can be difficult
    to see in bright sunlight. The speakers are also not very impressive,
    and the laptop gets noticeably warm during intensive workloads.

    Despite these issues, I think it offers good value for the price.
    It is a reliable option for students and professionals who need a
    laptop for regular productivity and moderate workloads.

    Review by Aditya Panwar.
    """)
print('summary: ',result['summary'])
print('sentiment: ',result['sentiment'])
print('pros: ',result['pros'])
print('cons: ',result['cons'])
print('name: ',result['name'])