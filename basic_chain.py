import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint



def get_model(model_name="llama2"):
    """Load model qua Ollama"""
    return Ollama(model=model_name)

def basic_chain(model=None, prompt=None):
    if not model:
        model = get_model()
    if not prompt:
        prompt = ChatPromptTemplate.from_template(
            "Tell me the most noteworthy books by the author {author}"
        )
    chain = prompt | model
    return chain


def main():
    load_dotenv()  # nếu bạn có file .env, nhưng Ollama thì không cần API token

    prompt = ChatPromptTemplate.from_template(
        "Tell me the most noteworthy books by the author {author}"
    )
    chain = basic_chain(prompt=prompt) | StrOutputParser()

    results = chain.invoke({"author": "William Faulkner"})
    print(results)


if __name__ == "__main__":
    main()
