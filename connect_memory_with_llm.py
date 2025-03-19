import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# step 1 : Setup LLM (Mistral with HuggingFace)

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(HUGGINGFACE_REPO_ID):
    llm = HuggingFaceEndpoint(
        repo_id = HUGGINGFACE_REPO_ID,
        temperature = 0.5,
        model_kwargs = {"token": HF_TOKEN,
                        "max_length": "512"}
    )
    return llm

# step 2 : Connect LLM with FAISS & Create chain



CUSTOM_PROMPT_TEMPLATE= """
Use the pieces of information provided in the context to answer user's question.
if you don't know answer, just say that you don't know, don't try to make up answer.
don't provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):
    prompt = PromptTemplate(template = CUSTOM_PROMPT_TEMPLATE, input_variables = ["context", "question"])
    return prompt

# LOAD DATABASE
DB_FAISS_PATH = "vector/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization = True)

# Create QA chain

qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(HUGGINGFACE_REPO_ID),
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs = {'k': 3}),
    return_source_documents = True,
    chain_type_kwargs = {'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query

user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])