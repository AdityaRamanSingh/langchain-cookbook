from langchain_core.prompts import PromptTemplate

# tempelate
template = PromptTemplate(
    template = """You are a research paper summarization assistant. Summarize the following research paper based on the user's requirements. Research Paper: {paper_input} Explanation Style: {style_input} Summary Length: {length_input} Provide a clear and accurate summary covering: 1. The main problem addressed 2. The proposed approach or methodology 3. Key findings or results 4. Important conclusions Keep the explanation in the requested style and length.""",
input_variables= ["paper_input", "style_input", "length_input"],

# validate_template validates if all innpute are present or if any extra is there
validate_template = True
)

template.save("template.json")