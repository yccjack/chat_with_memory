import asyncio
from pathlib import Path

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel, RunnablePassthrough, RunnableLambda
)
from langchain_core.runnables.history import RunnableWithMessageHistory


# 中文Token估算函数
def count_chinese_tokens(text: str) -> int:
    chinese_chars = sum('\u4e00' <= char <= '\u9fff' for char in text)
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.33 + other_chars * 0.25)


# 加载PDF并构建向量库
def init_knowledge_base(directory="docs"):
    documents = []
    for pdf_file in Path(directory).glob("*.pdf"):
        if pdf_file.is_file():
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents.extend(loader.load())
            except Exception as e:
                print(f"跳过文件 {pdf_file.name}，原因: {e}")
                continue

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=count_chinese_tokens
    )
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})


# 构建LangChain QA链
def create_qa_chain(retriever):
    llm = ChatOllama(model="deepseek-r1:14b")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个书籍内容分析助手，请根据提供的上下文内容来回答用户的问题。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "关于书籍内容的问题：{input}")
    ])

    def extract_text(input_dict):
        return input_dict["input"] if isinstance(input_dict, dict) else input_dict

    # 主 QA 任务链
    qa_chain = (
        RunnableParallel({
            "context": RunnablePassthrough() | RunnableLambda(extract_text) | retriever,
            "input": RunnablePassthrough() | RunnableLambda(extract_text),
            "chat_history": RunnablePassthrough().with_config(configurable={"key": "chat_history"})
        }).with_config(runnables_have_config=True)  # ✅ 关键修复
        | prompt
        | llm
        | StrOutputParser()
    )

    # 加入 Memory 支持
    return RunnableWithMessageHistory(
        qa_chain,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="input",
        history_messages_key="chat_history"
    )


# 异步对话主流程
async def chat_with_memory():
    retriever = init_knowledge_base()
    qa_chain = create_qa_chain(retriever)

    print("🤖 本地知识库问答机器人已启动，输入 '退出' 结束会话。")
    while True:
        try:
            user_input = await asyncio.to_thread(input, "你：")
            if user_input.lower().strip() in ('exit', 'quit', '退出'):
                break

            print("AI：", end="", flush=True)
            result = await qa_chain.ainvoke(
                {"input": user_input},
                config={"configurable": {"session_id": "fixed_session"}}
            )
            print(result.strip())

        except KeyboardInterrupt:
            print("\n对话中止")
            break


if __name__ == "__main__":
    asyncio.run(chat_with_memory())
