import os
from docx import Document

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# for costs calculations
# from langchain.callbacks import get_openai_callback


# OPENAI_API_KEY = os.environ['OPENAI_KEY']
fileName = "PythonExercise.docx" 


# read the text
def fileReader(fileName):
    doc = Document(fileName)
    text = ""
    paragraphs=[]
    for paragraph in doc.paragraphs:
        text += paragraph.text
        paragraphs.append(paragraph.text)
    return text, paragraphs

# split into chunks with context
def textSpliter(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )   
    return text_splitter.split_text(text)

# create embeddings
def translateDataToKnowledge(chunksOfText):
      embeddings = OpenAIEmbeddings()
      return FAISS.from_texts(chunksOfText, embeddings)

def getOpenAiAnswer(knowledge, question, chain):
    # semantic search in the knowledge base
    docs = knowledge.similarity_search(question)
    # GPT answer
    return chain.run(input_documents=docs, question=question)


text, paragraphs = fileReader(fileName)
cntxText = textSpliter(text)
knowledge = translateDataToKnowledge(cntxText)

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")


answer = getOpenAiAnswer(knowledge, question, chain)