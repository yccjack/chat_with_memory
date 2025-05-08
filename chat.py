from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
import asyncio
from langchain_community.vectorstores import FAISS
from pathlib import Path  # 添加到文件顶部导入部分
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pdfminer.pdffont import PDFFont
from langchain_huggingface import HuggingFaceEmbeddings
# 使用新版LCEL语法替代LLMChain
from langchain_core.runnables import RunnablePassthrough
# 确保使用完整导入路径
from langchain_core.output_parsers import StrOutputParser


# 中文 token 近似计算函数
def count_chinese_tokens(text: str) -> int:
    chinese_chars = sum('\u4e00' <= char <= '\u9fff' for char in text)
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.33 + other_chars * 0.25)


# 修改FontBox解析逻辑（需继承PDFFont类）
class SafePDFFont(PDFFont):
    def get_bbox(self):
        try:
            return super().get_bbox()
        except:
            return (0, 0, 1000, 1000)  # 默认bbox值


def validate_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            return f.read(4) == b'%PDF'
    except:
        return False


async def validate_answer(context, answer):
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    template = """判断以下内容是否来自给定上下文：
    上下文：{context}
    回答：{answer}
    只需输出True/False"""

    # 验证链改用RunnableSequence
    validation_chain = (
            RunnablePassthrough()
            | prompt
            | llm
            | StrOutputParser()
    )
    return await validation_chain.arun(context=context, answer=answer)


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


prompt = ChatPromptTemplate.from_messages([
    ("system", """你只能根据以下上下文回答问题：
{context}
---
回答规则：
1. 如果问题与上下文无关，回答："根据知识库无法回答该问题"
2. 回答必须直接引用上下文原文
3. 禁止添加任何非上下文的信息"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


# 文档加载与处理
def init_knowledge_base(directory="docs"):
    documents = []
    for pdf_file in Path(directory).glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"跳过文件 {pdf_file.name} 因加载错误: {str(e)}")
            continue

    # 文档切片（建议参数）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=count_chinese_tokens  # 复用您的token计数函数
    )
    splits = text_splitter.split_documents(documents)

    # 修改初始化Embeddings的代码
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},  # 如果没有GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 3,  # 返回最相关的3个片段
            "score_threshold": 0.7  # 相似度阈值
        }
    )


def create_qa_chain(retriever):
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    # 正确的LLMChain初始化方式
    validation_prompt = PromptTemplate(
        input_variables=["context", "answer"],
        template="判断回答是否来自上下文：\n上下文：{context}\n回答：{answer}\n只需输出True/False"
    )
    validation_chain = LLMChain(
        llm=llm,
        prompt=validation_prompt
    )

    # 原有检索链
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
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
