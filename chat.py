from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 中文 token 近似计算函数
def count_chinese_tokens(text: str) -> int:
    chinese_chars = sum('\u4e00' <= char <= '\u9fff' for char in text)
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.33 + other_chars * 0.25)


# 初始化模型
llm = ChatOllama(model="deepseek-r1:14b")

# 初始化摘要记忆（使用自定义 token 计数器）
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=300,
    return_messages=True,
    chat_memory=ChatMessageHistory(),
    get_num_tokens=lambda x: count_chinese_tokens(x)  # 替代默认的 tokenizer
)

# # Prompt 模板
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一个专业的助手"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}")
# ])


# 修改原有prompt模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下上下文回答问题：\n{context}\n---"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


# 文档加载与处理
def init_knowledge_base(directory="docs"):
    loader = DirectoryLoader(directory, glob="**/*.pdf")  # 支持PDF/TXT等格式
    documents = loader.load()

    # 文档切片（建议参数）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=count_chinese_tokens  # 复用您的token计数函数
    )
    splits = text_splitter.split_documents(documents)

    # 创建向量存储（使用本地Embedding）
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore.as_retriever()


# 创建检索链
def create_qa_chain(retriever):
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain

    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)


async def chat_with_memory():
    # 初始化知识库
    retriever = init_knowledge_base()
    qa_chain = create_qa_chain(retriever)

    while True:
        try:
            user_input = await asyncio.to_thread(input, "你：")
            if user_input.lower() in ('exit', 'quit', '退出'):
                break

            print("AI：", end="", flush=True)

            # 获取历史记录
            memory_variables = memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])

            # 执行检索增强生成
            result = await qa_chain.ainvoke({
                "input": user_input,
                "chat_history": chat_history
            })

            print(result["answer"], end="", flush=True)
            memory.save_context({"input": user_input}, {"output": result["answer"]})

        except KeyboardInterrupt:
            print("\n对话终止")
            break


# 启动
if __name__ == "__main__":
    asyncio.run(chat_with_memory())
