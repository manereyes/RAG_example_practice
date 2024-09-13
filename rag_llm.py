### The goal is creating an API capable of insert and list embeddings inside ChromaDB ###
from langchain_community.document_loaders import CSVLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate


### Load Dataset ###
dataset = './data/pokemon.csv'
loader = CSVLoader(file_path=dataset)
docs = loader.load()
#print(docs[0].page_content)

### Load LLM ###
llm = OllamaLLM(
    model="llama3.1"
)

### Load Embeddings ###
embeddings = OllamaEmbeddings(
    model="llama3.1"
)
#print(embeddings)

### Text Splitter - to reduce text into chunks of data.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
#print(len(splits))
#print(splits)
### Load vector store with Choma ###
vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)

### Build and load retriever ###
retriever = vector_store.as_retriever()

### Set temp and query ###
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is Pikachu's type?")
print(response)