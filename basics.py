import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Api key not set.")

model = init_chat_model("gpt-4o-mini", model_provider="openai")


# 1. way to create a prompt
messages = [
    SystemMessage("Translate the following from English to Hungarian."),
    HumanMessage("Hi everybody!"),
]

# 2. way to create a prompt ( structured )
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Hungarian", "text": "How are you?"})

# 1. way to get response
""" response = model.invoke(prompt)
print(response) """

# 2. way to get response ( streaming )
""" for token in model.stream(messages):
    print(token.content, end="") """


from langchain_community.document_loaders import PyPDFLoader

file_path = "opsys.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

""" print(len(docs))
print(docs[0].page_content) """

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, chunk_overlap=100, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

# print(len(all_splits))

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_1 = embeddings.embed_query(all_splits[0].page_content)

# print(vector_1)

from langchain_core.vectorstores import InMemoryVectorStore

""" vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits) """

# print(ids)

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("test-index")

vector_store = PineconeVectorStore(embedding=embeddings, index=index)

ids = vector_store.add_documents(documents=all_splits)

# print(ids)

# result = vector_store.similarity_search("Mi az a VFS?")

from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


# print(retriever.batch(["Mi az a VFS?", "Mi az a NFS?"]))

from typing import Optional
from pydantic import BaseModel, Field


class Subject(BaseModel):
    """Information about a subject in the PDF."""

    title: Optional[str] = Field(default=None, description="The title of the subject.")
    keyWords: Optional[List[str]] = Field(
        default=None,
        description="All the key words corresponding to the title and the subject.",
    )


class Data(BaseModel):
    """Extracted data about subject."""

    subjects: List[Subject]


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

structured_llm = model.with_structured_output(schema=Data)
text = vector_store.similarity_search("Mi az a VFS?")
prompt = prompt_template.invoke({"text": text[0]})
# print(structured_llm.invoke(prompt))
