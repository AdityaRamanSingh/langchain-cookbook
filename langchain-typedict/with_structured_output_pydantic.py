
# Import Hugging Face classes to connect to the LLM
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Import Pydantic classes to define and validate the output structure
from pydantic import BaseModel, Field

# Import typing helpers
from typing import Optional, Literal

# Import dotenv to load environment variables from the .env file
from dotenv import load_dotenv

# Import LangChain components for prompts and output parsing
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables, such as your Hugging Face token
load_dotenv()


# --------------------------------------------------
# STEP 1: Initialize the Hugging Face LLM
# --------------------------------------------------

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

# Wrap the LLM in ChatHuggingFace so it can work with chat messages
model = ChatHuggingFace(llm=llm)


# --------------------------------------------------
# STEP 2: Define the Pydantic output schema
# --------------------------------------------------

class Review(BaseModel):
    # Extract the main topics discussed in the review
    key_themes: list[str] = Field(
        description="List all key themes in the review"
    )

    # Return a single string summarizing the review
    summary: str = Field(
        description="A brief summary of the review"
    )

    # Restrict sentiment to these three possible values
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the review"
    )

    # Store multiple advantages; default to None if unavailable
    pros: Optional[list[str]] = Field(
        default=None,
        description="List the advantages mentioned in the review"
    )

    # Store multiple disadvantages; default to None if unavailable
    cons: Optional[list[str]] = Field(
        default=None,
        description="List the disadvantages mentioned in the review"
    )

    # Store the reviewer's name as a single string
    name: Optional[str] = Field(
        default=None,
        description="Name of the reviewer, if provided"
    )


# --------------------------------------------------
# STEP 3: Create a Pydantic output parser
# --------------------------------------------------

# This parser converts the LLM's JSON response into a Review object
# and validates the output against the schema defined above
parser = PydanticOutputParser(pydantic_object=Review)


# --------------------------------------------------
# STEP 4: Create the prompt template
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        # Tell the model what to do and provide the required output format
        "Analyze the customer review and extract the requested information.\n"
        "{format_instructions}"
    ),
    (
        "human",
        # The actual customer review will be inserted here
        "{review}"
    )
])


# --------------------------------------------------
# STEP 5: Connect the components into a LangChain chain
# --------------------------------------------------

# Flow: Prompt -> LLM -> Parser
# The output of each component becomes the input of the next
chain = prompt | model | parser


# --------------------------------------------------
# STEP 6: Provide the customer review
# --------------------------------------------------

review_text = """
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
"""


# --------------------------------------------------
# STEP 7: Invoke the chain
# --------------------------------------------------

# Pass the review and the format instructions to the prompt
result = chain.invoke({
    "review": review_text,

    # Automatically generate instructions from the Pydantic schema
    "format_instructions": parser.get_format_instructions()
})


# --------------------------------------------------
# STEP 8: Print the structured output
# --------------------------------------------------

# The result is a validated Review object, not a plain dictionary

print("\n--- Complete Review Analysis ---")
print(result)

print("\n--- Individual Fields ---")
print("Key Themes:", result.key_themes)
print("Summary:", result.summary)
print("Sentiment:", result.sentiment)
print("Pros:", result.pros)
print("Cons:", result.cons)
print("Reviewer Name:", result.name)