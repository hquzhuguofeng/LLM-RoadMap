from pymilvus import MilvusClient, DataType, Function, FunctionType
from openai import OpenAI
import json



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 搜索和查询
# 1、大模型初始化
openai_client = OpenAI(
	base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
	api_key="sk-25be5a383bd8450eb701bd44da3a8cd2"
)

# 2、实例化Milvus客户端对象
client = MilvusClient(
    uri="http://localhost:19530",
    db_name="milvus_database"
)

# 3、定义文本embedding处理函数
def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-v4",dimensions=1536)
        .data[0]
        .embedding
    )

# # 4、ANN搜索
# # 近似近邻（ANN）搜索以记录向量嵌入排序顺序的索引文件为基础
# # 根据接收到的搜索请求中携带的查询向量查找向量嵌入子集，将查询向量与子群中的向量进行比较，并返回最相似的结果
# question = "时序增强关系敏感知识迁移"
# res = client.search(
#     collection_name="my_collection_demo_chunked",
#     anns_field="content_dense",
#     data=[emb_text(question)],
#     limit=3,
#     search_params={"metric_type": "COSINE"},
#     output_fields=["title", "content_chunk", "link", "pubAuthor"]
# )
# retrieved_lines_with_distances = [
#     (res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
# ]
# print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

# # 5、使用标准过滤再进行ANN搜索
# # 若集合中同时包含向量嵌入及其元数据，可以在 ANN 搜索之前过滤元数据，以提高搜索结果的相关性
# # 过滤符合搜索请求中过滤条件的实体，在过滤后的实体中进行 ANN 搜索
# question = "时序增强关系敏感知识迁移"
# res = client.search(
#     collection_name="my_collection_demo_chunked",
#     anns_field="content_dense",
#     data=[emb_text(question)],
#     limit=3,
#     filter='pubAuthor like "机器之心%"',
#     # filter='pubAuthor like "量子位%"',
#     search_params={"metric_type": "COSINE"},
#     output_fields=["title", "content_chunk", "link", "pubAuthor"]
# )
# retrieved_lines_with_distances = [
#     (res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
# ]
# print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

# # 6、使用迭代过滤再进行ANN搜索
# # 标准过滤过程能有效地将搜索范围缩小到很小的范围。但是，过于复杂的过滤表达式可能会导致非常高的搜索延迟
# # 使用迭代过滤的搜索以迭代的方式执行向量搜索。迭代器返回的每个实体都要经过标量过滤，这个过程一直持续到达到指定的 topK 结果为止
# question = "时序增强关系敏感知识迁移"
# res = client.search(
#     collection_name="my_collection_demo_chunked",
# 	anns_field="content_dense",
#     data=[emb_text(question)],
#     limit=5,
#     filter='pubAuthor like "机器之心%"',
#     output_fields=["title", "content_chunk", "link", "pubAuthor"],
#     search_params={
# 		"metric_type": "COSINE",
#         "hints": "iterative_filter"
#     }
# )
# retrieved_lines_with_distances = [
#     (res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
# ]
# print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

# 7、范围搜索
# 执行范围搜索请求时，Milvus 以 ANN 搜索结果中与查询向量最相似的向量为圆心，以搜索请求中指定的半径为外圈半径，以range_filter为内圈半径，画出两个同心圆
# 所有相似度得分在这两个同心圆形成的环形区域内的向量都将被返回
# 这里，range_filter可以设置为0，表示将返回指定相似度得分（半径）范围内的所有实体
# question = "时序增强关系敏感知识迁移"
# res = client.search(
#     collection_name="my_collection_demo_chunked",
# 	anns_field="content_dense",
#     data=[emb_text(question)],
#     limit=3,
#     output_fields=["title", "content_chunk", "link", "pubAuthor"],
#     search_params={
# 		"metric_type": "COSINE",
#         "params": {
#             "radius": 0.2,
#             "range_filter": 1
#         }
#     }
# )
# retrieved_lines_with_distances = [
#     (res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
# ]
# print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

# # 8、分组搜索
# # 分组搜索允许 Milvus 根据指定字段的值对搜索结果进行分组，以便在更高层次上汇总数据
# # 根据提供的查询向量执行 ANN 搜索，找到与查询最相似的所有实体，按指定的group_by_field 对搜索结果进行分组
# # 根据limit参数的定义，返回每个组的顶部结果，并从每个组中选出最相似的实体
# question = "时序增强关系敏感知识迁移"
# res = client.search(
# 	collection_name="my_collection_demo_chunked",
# 	anns_field="content_dense",
# 	data=[emb_text(question)],
#     limit=3,
#     group_by_field="pubAuthor",
#     output_fields=["title", "content_chunk", "link", "pubAuthor"]
# )
# retrieved_lines_with_distances = [
#     (res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
# ]
# print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

# # 9、获取查找持有指定主键的实体
# res = client.get(
#     collection_name="my_collection_demo_chunked",
#     ids=[460348642159470562, 460348642159470563, 460348642159470564],
#     output_fields=["title", "content_chunk", "link", "pubAuthor"]
# )
# print(f"res:{res}")

# # 10、查询通过自定义过滤条件查找实体时，请使用查询方法
res = client.query(
    collection_name="my_collection_demo_chunked",
    filter='pubAuthor like "量子位%"',
    output_fields=["title", "content_chunk", "link", "pubAuthor"],
    limit=2
)
print(f"res:{res}")

