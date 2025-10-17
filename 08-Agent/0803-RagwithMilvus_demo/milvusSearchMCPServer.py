import logging
from mcp.server.lowlevel import Server
from mcp.types import Resource, Tool, TextContent
from mixTextSearch import MilvusSearchManager
from dotenv import load_dotenv
import os



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 日志相关配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_mcp_server")

# 从环境变量获取配置参数
load_dotenv()

# 实例化Server
mcp = Server("rag_mcp_server")


# 声明 list_tools 函数为一个列出工具的接口
# 列出可用的 MySQL 工具
@mcp.list_tools()
async def list_tools() -> list[Tool]:
    logger.info("Listing tools...")
    # 函数返回一个列表，其中包含一个 Tool 对象
    # 每个 Tool 对象代表一个工具，其属性定义了工具的功能和输入要求
    return [
        Tool(
            # 工具的名称
            name="search_documents",
            # 工具的描述
            description="执行文档搜索",
            # 定义了工具的输入模式（Schema），用于描述输入数据的格式和要求
            inputSchema={
                # 定义输入为一个 JSON 对象
                "type": "object",
                # 定义输入对象的属性
                "properties": {
                    # 指明此属性存储要执行搜索的内容
                    "query_text": {
                        "type": "string",
                        "description": "执行搜索的内容"
                    },
                    # 指明此属性存储要执行的过滤条件内容
                    "filter_query": {
                        "type": "string",
                        "default": "##None##",
                        "description": "过滤条件的自然语言描述内容,默认值为##None##。如:文章发布时间在2025年9月3号到5号之间的文章,作者是新智元的文档"
                    },
                    # 指明此属性存储要执行搜索的类型
                    "search_type": {
                        "type": "string",
                        "default": "hybrid",
                        "description": "可选 dense、sparse、hybrid，其中dense为语义搜索、sparse为全文搜索或关键词搜索、hybrid为混合搜索，默认为hybrid"
                    },
                    # 指明此属性存储要执行搜索返回结果的数量
                    "limit": {
                        "type": "number",
                        "default": 2,
                        "description": "结果返回的数量,默认值为2"
                        # "description": "Number of results,default 2"
                    }
                },
                # 列出输入对象的必需属性
                "required": ["query_text","filter_query","search_type","limit"]
            }
        )
    ]


# 声明 call_tool 函数为一个工具调用的接口
# 根据传入的工具名称和参数执行相应的搜索
# name: 工具的名称（字符串），指定要调用的工具
# arguments: 一个字典，包含工具所需的参数
@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # 检查工具名称 name 是否是 search_documents
    # 如果 query_text 为空或未提供，抛出 ValueError 异常，提示用户必须提供查询语句
    if name != "search_documents":
        raise ValueError(f"Unknown tool: {name}")

    query_text = arguments.get("query_text")
    search_type = arguments.get("search_type")
    limit = arguments.get("limit")
    filter_query = arguments.get("filter_query")
    if not query_text:
        raise ValueError("Query is required")
    if not filter_query:
        raise ValueError("filter_query is required")
    if not search_type:
        raise ValueError("Search type is required")
    if not limit:
        raise ValueError("Limit is required")

    try:
        # 初始化搜索管理器
        search_manager = MilvusSearchManager(
            milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
            db_name=os.getenv("MILVUS_DATABASE_NAME", "milvus_database"),
            openai_base_url=os.getenv("LLM_BASE_URL", "https://nangeai.top/v1"),
            openai_api_key=os.getenv("LLM_API_KEY", "")
        )

        # 执行混合搜索示例
        filter_result = search_manager.search_with_filter(
            collection_name="my_collection_demo_chunked",
            query_text=query_text,
            filter_query=filter_query,
            search_type=search_type,
            limit=limit
        )
        if filter_result["success"]:
            print(f"✅ 过滤搜索成功")
            print(f"结果数量: {filter_result['total_results']}")

            if filter_result["results"] and len(filter_result["results"]) > 0:
                filtered_items = [
                    (
                        res.entity.get("title", ""),
                        res.entity.get("content_chunk", ""),
                        res.entity.get("link", ""),
                        res.entity.get("pubAuthor", ""),
                        res.entity.get("pubDate", ""),
                        res.distance
                    ) for res in filter_result["results"][0]
                ]

                # 将过滤搜索结果拼接成字符串
                filtered_result_string = ""
                for idx, item in enumerate(filtered_items, 1):
                    title, content_chunk, link, pubAuthor, pubDate, distance = item
                    record = (
                        f"过滤文章{idx}:\n"
                        f"文章标题: {title}\n"
                        f"文章内容片段: {content_chunk[:100]}...\n"
                        f"文章原始链接: {link}\n"
                        f"文章发布者: {pubAuthor}\n"
                        f"文章发布时间: {pubDate}\n"
                    )
                    filtered_result_string += record
                print(f"过滤搜索结果:\n{filtered_result_string}")
                # 返回一个包含查询结果的 TextContent 对象
                return [TextContent(type="text", text=filtered_result_string)]
        else:
            print(f"❌ 过滤搜索失败: {filter_result['error']}")
            if "suggestions" in filter_result:
                print(f"   建议查询: {filter_result['suggestions']}")
            # 返回一个包含查询结果的 TextContent 对象
            return [TextContent(type="text", text="\n未检索到相关结果")]

    except Exception as e:
        logger.error(f"主程序执行异常: {e}")
        print("程序异常终止。")
        return [TextContent(type="text", text="\n主程序执行异常，程序异常终止")]

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="streamable_http")