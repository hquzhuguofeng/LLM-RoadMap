from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import asyncio



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


async def run():
    # 创建与服务器的SSE连接，并返回 read_stream 和 write_stream 流
    async with streamablehttp_client(url="http://127.0.0.1:8010/mcp") as (read_stream, write_stream, get_session_id_callback):
        # # 创建一个客户端会话对象，通过 read_stream 和 write_stream 流与服务器交互
        async with ClientSession(read_stream, write_stream) as session:
            # 向服务器发送初始化请求，确保连接准备就绪
            # 建立初始状态，并让服务器返回其功能和版本信息
            capabilities = await session.initialize()
            print(f"Supported capabilities:{capabilities.capabilities}/n/n")

            # 获取可用的工具列表
            tools = await session.list_tools()
            print(f"Supported tools:{tools}/n/n")
            # with open("output.txt", 'w', encoding='utf-8') as file:
            #     file.write(str(tools))

            # 工具功能测试
            result = await session.call_tool("search_documents",{"query_text":"全球AI百强榜发布,第一是谁？"})
            print(f"Supported result:{result}")


if __name__ == "__main__":
    # 使用 asyncio 启动异步的 run() 函数
    asyncio.run(run())