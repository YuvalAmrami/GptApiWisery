import os

from docx import Document

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI



# for costs calculations
# from langchain.callbacks import get_openai_callback

############################################# setting up the AI:
fileName = "PythonExercise.docx" 

###################### AI functions:
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
      return FAISS.from_texts(chunksOfText, embeddings)

def getOpenAiAnswer(knowledge, question, chain):

    # print(question)
    # semantic search in the knowledge base
    docs = knowledge.similarity_search(question)
    # GPT answer
    return chain.run(input_documents=docs, question=question)

# loading and setting the file:
embeddings = OpenAIEmbeddings()
text, paragraphs = fileReader(fileName)
cntxText = textSpliter(text)
knowledge = translateDataToKnowledge(cntxText)
retriever = knowledge.as_retriever()


# setting the AI system
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")
# template = "Answer the question based only on the following context."

############################# setting the API 

app = FastAPI()

class Question(BaseModel):
    question: str


@app.get("/")
def hello():
    return "hello"

@app.post("/")
def askQuestion(TheQuestion: Question):
    # print(TheQuestion.question)
    question = TheQuestion.question
    return getOpenAiAnswer(knowledge, question, chain)


