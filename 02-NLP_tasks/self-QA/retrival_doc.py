import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import json
from tqdm import tqdm

# 创建log日志器
import logging
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='test.log',
                    filemode='w')


def format_docs(docs):
    return "\n".join(
        [f"参考内容{i+1}\n{doc.page_content}\n" for i, doc in enumerate(docs)]
    )

os.environ["OPENAI_API_KEY"] = "None"
emb_model = "D:/pretrained_model/bge-large-zh-v1.5"
embedding_model_kwargs = {'device': 'cpu'}
embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
)


template = """你是一名根据参考内容回答用户问题的机器人，你的职责是：根据提供的参考内容回答用户的问题。如果参考内容与问题不相关，你可以选择忽略参考内容，只回答问题。

##参考内容：
{context}

##用户问题：
{question}

请根据参考内容，回答用户问题，回答内容要简洁，说重点，不要做过多的解释，输出内容限制在200字符内。
"""


def get_chunk_retrival(text, file_path):
    documents = [Document(
                    page_content=text,
                    metadata={"source": file_path},
                )]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)

    retriver = db.as_retriever(search_kwargs={"k": 3})

    return retriver

def get_all_file_name(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_list.append(file.split(".")[0])
    return file_list








