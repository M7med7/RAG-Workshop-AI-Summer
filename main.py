from typing import List, Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

from langgraph.graph import START, StateGraph

llm = init_chat_model(model="smollm2:135m", model_provider="ollama")
embedding = OllamaEmbeddings(model="granite-embedding:30m")
vector_db = Chroma(
    collection_name="test",
    embedding_function=embedding,
)

loader = ArxivLoader(query="Transformers", load_max_docs=150)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=256)
chunks = splitter.split_documents(docs)

vector_db.add_documents(chunks)

prompt = PromptTemplate(
    template="""
You are a helpful assistant. Answer the question based on the provided context.

question: {question}
context: {context}
""",
    input_variables=["question", "context"],
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retriever(state: State) -> dict[str, Any]:
    results = vector_db.similarity_search(state["question"], k=10)
    return {"context": results}

def generator(state: State) -> dict[str, Any]:
    joined = "\n\n".join(doc.page_content for doc in state["context"])
    prompt_value = prompt.format_prompt(question=state["question"], context=joined)
    response = llm.invoke(prompt_value.to_string())
    return {"answer": response}

graph_builder = StateGraph(State).add_sequence([retriever, generator])
graph_builder.add_edge(START, retriever, "retriever")
graph = graph_builder.compile()

out = graph.invoke({"question": "What is the Transformer architecture?"})
print(out["answer"])
