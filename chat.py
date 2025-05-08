import asyncio
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# 用于中文 token 估算
def count_chinese_tokens(text: str) -> int:
    chinese_chars = sum('\u4e00' <= char <= '\u9fff' for char in text)
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.3 + other_chars * 0.25)


# 加载 PDF 文档并建立向量数据库
def init_vectorstore():
    docs = []
    for pdf_path in Path("docs").glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=count_chinese_tokens
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return FAISS.from_documents(chunks, embeddings)


# 构造带记忆的 QA Chain
def build_chain(retriever):
    llm = ChatOllama(model="deepseek-r1:14b")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个基于书籍的中文知识助手，请根据上下文回答问题。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = (
        RunnableParallel({
            "question": lambda x: x["question"],
            "context": lambda x: retriever.invoke(x["question"]),
            "chat_history": lambda x: x["chat_history"],
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        lambda session_id: InMemoryChatMessageHistory(),
        input_messages_key="question",
        history_messages_key="chat_history"
    )


# 主循环
async def main():
    vectorstore = init_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chain = build_chain(retriever)

    print("输入 'exit' 可退出。")
    while True:
        user_input = await asyncio.to_thread(input, "\n你：")
        if user_input.lower() in {"exit", "quit", "退出"}:
            break

        response = await chain.ainvoke(
            {"question": user_input},
            config={"configurable": {"session_id": "default"}}
        )
        print("AI：", response)


if __name__ == "__main__":
    asyncio.run(main())
