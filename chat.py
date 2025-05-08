from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
import asyncio


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

# Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的助手"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


# 异步对话
async def chat_with_memory():
    chain = prompt | llm
    print("输入 'exit' 退出对话")

    while True:
        try:
            user_input = await asyncio.to_thread(input, "你：")
            if user_input.lower() in ('exit', 'quit', '退出'):
                break

            print("AI：", end="", flush=True)

            # 获取 chat_history，如果没有则创建一个空列表
            memory_variables = memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])

            async for chunk in chain.astream({
                "input": user_input,
                "chat_history": chat_history
            }):
                print(chunk.content, end="", flush=True)

            # 保存当前对话上下文
            memory.save_context({"input": user_input}, {"output": chunk.content})
            print()

        except KeyboardInterrupt:
            print("\n对话终止")
            break


# 启动
if __name__ == "__main__":
    asyncio.run(chat_with_memory())
