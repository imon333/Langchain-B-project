from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Post-processing steps
uppercase_output = RunnableLambda(lambda text: text.upper())
count_words = RunnableLambda(lambda text: f"Word count: {len(text.split())}\n{text}")

# Combine everything using LCEL (| operator)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

if __name__ == "__main__":
    # Run the chain
    result = chain.invoke({"topic": "lawyers", "joke_count": 3})
    print(result)
