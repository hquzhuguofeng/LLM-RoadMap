from pymilvus import MilvusClient, DataType, Function, FunctionType
from openai import OpenAI
import json



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 全文搜索
# 1、大模型初始化
openai_client = OpenAI(
	base_url="https://nangeai.top/v1",
	api_key="sk-33RqjaXhsrjeapyKC3DiwvG1DdpMDU9tHCCmwVxnxcw5HmLS"
)

# 2、实例化Milvus客户端对象
client = MilvusClient(
    uri="http://localhost:19530",
    db_name="milvus_database"
)

# 3、定义文本embedding处理函数
def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

# 4、全文搜索分词测试
# 通用分词分析器
# analyzer_params = {"tokenizer": "standard", "filter": ["lowercase"]}
# analyzer_params = {"type": "standard"}
# 适合中文的分词分析器
# analyzer_params = {"tokenizer": "jieba", "filter": ["cnalphanumonly"]}
analyzer_params = {"type": "chinese"}
# 分词测试
# text = "An efficient system relies on a robust analyzer to correctly process text for various applications."
# text = "We-Math 2.0：全新多模态数学推理数据集 × 首个综合数学知识体系"
# text = "AI大咖齐聚！共议「人工智能+」国家战略落地路径"
text = "手机端大模型如何平衡性能和效率"
result = client.run_analyzer(
    text,
    analyzer_params
)
print(f"result:{result}")

# # 5、全文搜索
# # 全文搜索是一种通过匹配文本中特定关键词或短语来检索文档的传统方法
# # 它根据术语频率等因素计算出的相关性分数对结果进行排序
# # 语义搜索更善于理解含义和上下文，而全文搜索则擅长精确的关键词匹配，因此是语义搜索的有益补充
# # 对title进行全文搜索
# question = "AI智能体"
# res = client.search(
#     collection_name="my_collection_demo_chunked",
#     anns_field="title_sparse",
#     data=[question],
#     limit=3,
#     search_params={
#         'params': {'drop_ratio_search': 0.2},
#     },
#     output_fields=["title", "content_chunk", "link", "pubAuthor"]
# )
# retrieved_lines_with_distances = [
#     (res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
# ]
# print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

# # 对content进行全文搜索
# question = "AI智能体"
# res = client.search(
#     collection_name="my_collection_demo_chunked",
#     anns_field="content_sparse",
#     data=[question],
#     limit=3,
#     search_params={
#         'params': {'drop_ratio_search': 0.2},
#     },
#     output_fields=["title", "content_chunk", "link", "pubAuthor"]
# )
# retrieved_lines_with_distances = [
#     (res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
# ]
# print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

