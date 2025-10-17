from pymilvus import MilvusClient, DataType, Function, FunctionType
from pymilvus import AnnSearchRequest
from openai import OpenAI
from typing import List, Dict, Any, Optional, Callable, Union
from dotenv import load_dotenv
import os
import json
import logging
import time
import random
import re
from enum import Enum

# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 从环境变量获取配置参

load_dotenv()


class FilterOperator(Enum):
    """过滤操作符枚举"""
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not in"
    LIKE = "like"
    AND = "and"
    OR = "or"
    NOT = "not"


class MilvusFilterExpressionGenerator:
    """Milvus过滤表达式生成器"""

    def __init__(self, openai_client: OpenAI):
        """
        初始化过滤表达式生成器

        Args:
            openai_client: OpenAI客户端实例
        """
        self.openai_client = openai_client

        # 定义集合schema信息，用于验证和提示
        self.schema_info = {
            "fields": {
                "doc_id": {"type": "VARCHAR", "max_length": 100, "description": "文档唯一标识"},
                "chunk_index": {"type": "INT64", "description": "文档块序号"},
                "title": {"type": "VARCHAR", "max_length": 1000, "description": "文章标题"},
                "link": {"type": "VARCHAR", "max_length": 500, "description": "文章链接"},
                "pubDate": {"type": "VARCHAR", "max_length": 100,
                            "description": "发布时间，格式如'2025.08.28 08:55:00'"},
                "pubAuthor": {"type": "VARCHAR", "max_length": 100, "description": "发布者"},
                "content_chunk": {"type": "VARCHAR", "max_length": 3000, "description": "文档内容块"},
                "full_content": {"type": "VARCHAR", "max_length": 20000, "description": "完整文档内容"}
            },
            "operators": {
                "VARCHAR": ["==", "!=", "in", "not in", "like"],
                "INT64": ["==", "!=", ">", ">=", "<", "<=", "in", "not in"],
                "FLOAT": ["==", "!=", ">", ">=", "<", "<="]
            }
        }

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return f"""你是一个专业的Milvus过滤表达式生成器。你需要根据用户的自然语言查询，生成符合Milvus语法的过滤表达式。

集合Schema信息：
{json.dumps(self.schema_info, indent=2, ensure_ascii=False)}

Milvus过滤表达式语法规则：
1. 字符串字段需要用双引号包围，如: title == "人工智能"
2. 数值字段不需要引号，如: chunk_index > 5
3. 支持的操作符：==, !=, >, >=, <, <=, in, not in, like
4. 逻辑操作符：and, or, not
5. like操作符支持通配符%，如: title like "%AI%"
6. in操作符语法：field in ["value1", "value2"]
7. 日期比较需要转换为字符串格式进行比较
8. 不支持length()等函数，请使用字段直接比较
9. 字段名必须完全匹配schema中定义的字段名

重要注意事项：
- 所有字符串比较都必须使用双引号
- 不要使用未定义的字段名（如r，应该是pubAuthor）
- 不要使用不支持的函数（如length()）
- 确保操作符与字段类型匹配
- 如果用户的输入中有超过schema中定义的字段，请忽略多余的字段

示例：
- 用户："查找标题包含人工智能的文档" → title like "%人工智能%"
- 用户："查找作者是张三或李四的文档" → pubAuthor in ["张三", "李四"]
- 用户："查找2024年8月发布的文档" → pubDate like "2024.08%"
- 用户："查找标题包含AI且作者不是机器之心的文档" → title like "%AI%" and pubAuthor != "机器之心"
- 用户："查找作者是新智元的文档" → pubAuthor == "新智元"
- 用户："查找文档块序号大于5的内容" → chunk_index > 5
- 用户："查找2026年11月前发布的文章" → pubDate < "2026.11"


请只返回过滤表达式，不要包含其他解释。如果无法生成有效表达式，返回空字符串。"""

    def generate_filter_expression(self, user_query: str, max_retries: int = 3) -> str:
        """
        根据用户查询生成过滤表达式

        Args:
            user_query: 用户自然语言查询
            max_retries: 最大重试次数

        Returns:
            str: 生成的过滤表达式
        """
        if not user_query or not isinstance(user_query, str):
            logger.warning("用户查询为空或非字符串类型")
            return ""

        for attempt in range(max_retries):
            try:
                logger.debug(f"正在生成过滤表达式 (尝试 {attempt + 1}/{max_retries})")

                response = self.openai_client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "qwen-max-latest"),
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": f"用户查询：{user_query}"}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )

                filter_expr = response.choices[0].message.content.strip()

                # 验证生成的表达式
                if self._validate_filter_expression(filter_expr):
                    logger.debug(f"成功生成过滤表达式: {filter_expr}")
                    return filter_expr
                else:
                    logger.warning(f"生成的表达式验证失败: {filter_expr}")
                    if attempt == max_retries - 1:
                        return ""

            except Exception as e:
                logger.error(f"生成过滤表达式失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return ""

        return ""

    def _validate_filter_expression(self, expression: str) -> bool:
        """
        验证过滤表达式的有效性

        Args:
            expression: 过滤表达式

        Returns:
            bool: 表达式是否有效
        """
        if not expression:
            return False

        try:
            # 基本语法检查
            # 检查是否包含有效的字段名
            field_names = list(self.schema_info["fields"].keys())
            has_valid_field = any(field in expression for field in field_names)

            if not has_valid_field:
                logger.warning("表达式中未包含有效的字段名")
                return False

            # 检查括号匹配
            if expression.count('(') != expression.count(')'):
                logger.warning("表达式中括号不匹配")
                return False

            # 检查引号匹配
            if expression.count('"') % 2 != 0:
                logger.warning("表达式中引号不匹配")
                return False

            # 检查是否包含危险字符
            dangerous_chars = [';', '--', '/*', '*/', 'DROP', 'DELETE', 'UPDATE', 'INSERT']
            if any(char in expression.upper() for char in dangerous_chars):
                logger.warning("表达式包含危险字符")
                return False

            logger.debug("过滤表达式验证通过")
            return True

        except Exception as e:
            logger.error(f"验证过滤表达式时出错: {e}")
            return False


class MilvusSearchManager:
    """Milvus混合搜索管理器，提供多种搜索模式"""

    def __init__(self,
                 milvus_uri: str,
                 db_name: str,
                 openai_base_url: str,
                 openai_api_key: str):
        """
        初始化搜索管理器

        Args:
            milvus_uri: Milvus服务器地址
            db_name: 数据库名称
            openai_base_url: OpenAI API基础URL
            openai_api_key: OpenAI API密钥
        """
        self.milvus_uri = milvus_uri
        self.db_name = db_name
        self.milvus_client = None
        self.openai_client = None
        self.filter_generator = None

        # 初始化客户端
        self._init_clients(openai_base_url, openai_api_key)

    def _init_clients(self, openai_base_url: str, openai_api_key: str) -> None:
        """
        初始化Milvus和OpenAI客户端

        Args:
            openai_base_url: OpenAI API基础URL
            openai_api_key: OpenAI API密钥
        """
        try:
            # 验证参数
            if not self.milvus_uri or not isinstance(self.milvus_uri, str):
                raise ValueError("Milvus URI不能为空且必须是字符串类型")

            if not self.db_name or not isinstance(self.db_name, str):
                raise ValueError("数据库名称不能为空且必须是字符串类型")

            if not openai_api_key or not isinstance(openai_api_key, str):
                raise ValueError("OpenAI API密钥不能为空且必须是字符串类型")

            logger.info("正在初始化客户端连接...")

            # 初始化OpenAI客户端
            self.openai_client = OpenAI(
                base_url=openai_base_url,
                api_key=openai_api_key
            )
            logger.info("OpenAI客户端初始化成功")

            # 初始化过滤表达式生成器
            self.filter_generator = MilvusFilterExpressionGenerator(self.openai_client)
            logger.info("过滤表达式生成器初始化成功")

            # 初始化Milvus客户端
            self.milvus_client = MilvusClient(
                uri=self.milvus_uri,
                db_name=self.db_name
            )

            # 测试Milvus连接
            collections = self.milvus_client.list_collections()
            logger.info(f"Milvus客户端初始化成功，当前集合数量: {len(collections)}")

        except Exception as e:
            logger.error(f"客户端初始化失败: {e}")
            raise

    def emb_text(self, text: str, model: str = "text-embedding-v4", max_retries: int = 3) -> List[float]:
        """
        生成文本的向量嵌入

        Args:
            text: 要嵌入的文本
            model: 使用的嵌入模型
            max_retries: 最大重试次数

        Returns:
            List[float]: 文本的向量嵌入
        """
        if not text or not isinstance(text, str):
            logger.warning("输入文本为空或非字符串类型，返回零向量")
            return [0.0] * 1536  # 返回默认维度的零向量

        # 文本长度检查和截断
        if len(text) > 8000:  # OpenAI embedding模型的大致限制
            logger.warning(f"文本长度 {len(text)} 超过限制，将截断到8000字符")
            text = text[:8000]

        for attempt in range(max_retries):
            try:
                logger.debug(f"正在为文本生成嵌入向量 (尝试 {attempt + 1}/{max_retries})")

                response = self.openai_client.embeddings.create(
                    input=text,
                    model=model,
                    dimensions=1536
                )

                embedding = response.data[0].embedding
                logger.debug(f"成功生成 {len(embedding)} 维向量")
                return embedding

            except Exception as e:
                logger.warning(f"生成嵌入向量失败 (尝试 {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    # 指数退避重试
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"生成嵌入向量最终失败，返回随机向量")
                    # 返回随机向量作为备选
                    return [random.random() for _ in range(1536)]

    def _validate_search_params(self, collection_name: str, query_text: str,
                                search_type: str, limit: int) -> bool:
        """
        验证搜索参数的有效性

        Args:
            collection_name: 集合名称
            query_text: 查询文本
            search_type: 搜索类型
            limit: 返回结果数量

        Returns:
            bool: 参数是否有效
        """
        try:
            # 验证集合名称
            if not collection_name or not isinstance(collection_name, str):
                logger.error("集合名称不能为空且必须是字符串类型")
                return False

            # 检查集合是否存在
            if not self.milvus_client.has_collection(collection_name):
                logger.error(f"集合 '{collection_name}' 不存在")
                return False

            # 验证查询文本
            if not query_text or not isinstance(query_text, str):
                logger.error("查询文本不能为空且必须是字符串类型")
                return False

            # 验证搜索类型
            valid_search_types = ["dense", "sparse", "hybrid"]
            if search_type not in valid_search_types:
                logger.error(f"搜索类型必须是 {valid_search_types} 中的一种")
                return False

            # 验证limit参数
            if not isinstance(limit, int) or limit <= 0:
                logger.error("limit必须是大于0的整数")
                return False

            if limit > 1000:  # 设置合理的上限
                logger.warning(f"limit值 {limit} 过大，建议不超过1000")

            return True

        except Exception as e:
            logger.error(f"验证搜索参数时出错: {e}")
            return False

    def _perform_sparse_search(self, collection_name: str, query_text: str,
                               limit: int, output_fields: List[str], filter_expr: str = None) -> List[Dict]:
        """
        执行稀疏向量搜索（BM25全文搜索）

        Args:
            collection_name: 集合名称
            query_text: 查询文本
            limit: 返回结果数量
            output_fields: 输出字段列表
            filter_expr: 过滤表达式

        Returns:
            List[Dict]: 搜索结果
        """
        try:
            logger.info("执行稀疏向量搜索（BM25全文搜索）")

            search_params = {
                'params': {'drop_ratio_search': 0.2},
            }

            res = self.milvus_client.search(
                collection_name=collection_name,
                anns_field="title_sparse",
                data=[query_text],
                limit=limit,
                search_params=search_params,
                filter=filter_expr,
                output_fields=output_fields
            )

            logger.info(f"稀疏向量搜索完成，返回 {len(res[0]) if res else 0} 个结果")
            return res

        except Exception as e:
            logger.error(f"稀疏向量搜索失败: {e}")
            return []

    def _perform_dense_search(self, collection_name: str, query_text: str,
                              limit: int, output_fields: List[str], filter_expr: str = None) -> List[Dict]:
        """
        执行密集向量搜索（语义搜索）

        Args:
            collection_name: 集合名称
            query_text: 查询文本
            limit: 返回结果数量
            output_fields: 输出字段列表
            filter_expr: 过滤表达式

        Returns:
            List[Dict]: 搜索结果
        """
        try:
            logger.info("执行密集向量搜索（语义搜索）")

            # 生成查询向量
            model = os.getenv("LLM_EMBEDDING_MODEL", "text-embedding-v4")
            query_vector = self.emb_text(query_text, model)

            res = self.milvus_client.search(
                collection_name=collection_name,
                anns_field="content_dense",
                data=[query_vector],
                limit=limit,
                search_params={"metric_type": "COSINE"},
                filter=filter_expr,
                output_fields=output_fields
            )

            logger.info(f"密集向量搜索完成，返回 {len(res[0]) if res else 0} 个结果")
            return res

        except Exception as e:
            logger.error(f"密集向量搜索失败: {e}")
            return []

    def _create_rrf_ranker(self, k: int = 100) -> Function:
        """
        创建RRF（互惠排名融合）排名器

        Args:
            k: RRF参数，用于调节排名融合的平滑度

        Returns:
            Function: RRF排名器函数
        """
        try:
            # RRF Ranker 专门设计用于混合搜索场景
            # 根据多个向量搜索路径的排名位置而不是原始相似度得分来平衡搜索结果
            rrf_ranker = Function(
                name="rrf",
                input_field_names=[],
                function_type=FunctionType.RERANK,
                params={
                    "reranker": "rrf",
                    "k": k
                }
            )
            logger.debug("RRF排名器创建成功")
            return rrf_ranker
        except Exception as e:
            logger.error(f"创建RRF排名器失败: {e}")
            raise

    def _create_weight_ranker(self, weights: List[float], norm_score: bool = True) -> Function:
        """
        创建加权排名器

        Args:
            weights: 各搜索路径的权重列表，权重值应在[0,1]范围内
            norm_score: 是否对原始分数进行归一化处理

        Returns:
            Function: 加权排名器函数
        """
        try:
            if not weights or not all(0 <= w <= 1 for w in weights):
                raise ValueError("权重列表不能为空，且所有权重应在[0,1]范围内")

            weight_ranker = Function(
                name="weight",
                input_field_names=[],
                function_type=FunctionType.RERANK,
                params={
                    "reranker": "weighted",
                    "weights": weights,
                    "norm_score": norm_score
                }
            )
            logger.debug("加权排名器创建成功")
            return weight_ranker
        except Exception as e:
            logger.error(f"创建加权排名器失败: {e}")
            raise

    def search_documents(self,
                         collection_name: str,
                         query_text: str,
                         search_type: str,
                         limit: int = 5,
                         filter_query: str = "##None##",
                         embedding_function: Optional[Callable[[str], List[float]]] = None
                         ) -> List[Dict[str, Any]]:
        """
        搜索文档，支持稀疏搜索、密集搜索和混合搜索，支持过滤功能

        Args:
            collection_name: Milvus集合名称
            query_text: 查询文本
            search_type: 搜索类型，"dense", "sparse"或"hybrid"
            limit: 返回结果数量
            filter_query: 自然语言过滤条件
            embedding_function: 生成查询向量的函数

        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        try:
            logger.info(f"开始执行 {search_type} 搜索，查询文本: '{query_text}', 返回数量: {limit}")
            if filter_query != "##None##":
                logger.info(f"过滤条件: '{filter_query}'")

            # 参数验证
            if not self._validate_search_params(collection_name, query_text, search_type, limit):
                logger.error("搜索参数验证失败")
                return []

            # 生成过滤表达式
            filter_expr = None
            if filter_query != "##None##":
                filter_expr = self.filter_generator.generate_filter_expression(filter_query)
                if filter_expr:
                    logger.info(f"生成的过滤表达式: {filter_expr}")
                else:
                    logger.warning("无法生成有效的过滤表达式，将忽略过滤条件")

            output_fields = ["title", "content_chunk", "link", "pubAuthor", "pubDate"]

            if search_type == "sparse":
                # 稀疏向量搜索（BM25全文搜索）
                return self._perform_sparse_search(collection_name, query_text, limit, output_fields, filter_expr)

            elif search_type == "dense":
                # 密集向量搜索（语义搜索）
                if embedding_function is None:
                    embedding_function = self.emb_text
                return self._perform_dense_search(collection_name, query_text, limit, output_fields, filter_expr)

            elif search_type == "hybrid":
                # 混合搜索
                if embedding_function is None:
                    embedding_function = self.emb_text

                # 创建混合搜索请求
                search_param_1 = {
                    "data": [embedding_function(query_text)],
                    "anns_field": "content_dense",
                    "param": {"nprobe": 10, "metric_type": "COSINE"},
                    "limit": min(2, limit),
                    "expr": filter_expr  # 添加过滤表达式
                }
                search_param_2 = {
                    "data": [query_text],
                    "anns_field": "title_sparse",
                    "param": {"drop_ratio_search": 0.2},
                    "limit": min(2, limit),
                    "expr": filter_expr  # 添加过滤表达式
                }
                request_1 = AnnSearchRequest(**search_param_1)
                request_2 = AnnSearchRequest(**search_param_2)

                # 选择排名器，默认使用RRF
                rrf_ranker = self._create_rrf_ranker(k=100)
                # weight_ranker = self._create_weight_ranker(weights=[0.9, 0.1], norm_score=True)

                res = self.milvus_client.hybrid_search(
                    collection_name=collection_name,
                    reqs=[request_1, request_2],
                    ranker=rrf_ranker,
                    # ranker=weight_ranker,
                    limit=limit,
                    output_fields=output_fields
                )
                logger.info(f"混合搜索完成，返回结果数: {len(res[0]) if res else 0}")
                return res

            else:
                logger.error(f"不支持的搜索类型: {search_type}")
                return []

        except Exception as e:
            logger.error(f"搜索出错: {e}")
            return []

    def search_with_filter(self,
                           collection_name: str,
                           query_text: str,
                           filter_query: str,
                           search_type: str = "hybrid",
                           limit: int = 5) -> Dict[str, Any]:
        """
        使用自然语言过滤条件进行搜索的便捷方法

        Args:
            collection_name: 集合名称
            query_text: 搜索查询文本
            filter_query: 自然语言过滤条件
            search_type: 搜索类型
            limit: 返回结果数量

        Returns:
            Dict[str, Any]: 搜索结果和元信息
        """
        try:
            # 执行搜索
            search_results = self.search_documents(
                collection_name=collection_name,
                query_text=query_text,
                search_type=search_type,
                limit=limit,
                filter_query=filter_query
            )

            return {
                "results": search_results,
                "success": True,
                "filter_query": filter_query,
                "total_results": len(search_results[0]) if search_results else 0
            }

        except Exception as e:
            logger.error(f"过滤搜索失败: {e}")
            return {
                "results": [],
                "success": False,
                "error": str(e),
                "filter_query": filter_query
            }


# 搜索测试
if __name__ == "__main__":
    try:
        # 初始化搜索管理器
        search_manager = MilvusSearchManager(
            milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
            db_name=os.getenv("MILVUS_DATABASE_NAME", "milvus_database"),
            openai_base_url=os.getenv("LLM_BASE_URL", "https://nangeai.top/v1"),
            openai_api_key=os.getenv("LLM_API_KEY", "")
        )

        print("=" * 80)
        print("Milvus搜索功能测试")
        print("=" * 80)

        print("\n【测试2: 带过滤条件的搜索】")
        filter_result = search_manager.search_with_filter(
            # 集合名称
            collection_name="my_collection_demo_chunked",
            # 搜索的内容
            query_text="时序增强关系敏感知识迁移",

            # 过滤条件
            filter_query="##None##",
            # filter_query="文章发布在2025年9月2号之前",
            # filter_query="文章发布在2025年9月3号到5号之间，发布者是新智元",
            # filter_query="文章发布在2025年9月3号到5号之间,发布者是机器之心",
            # filter_query="文章发布在2025.09.05之前，发布者是机器之心，价格不超过500元，字数在200字内，",

            # dense为语义搜索、sparse为全文搜索或关键词搜索、hybrid为混合搜索
            search_type="hybrid",
            # 返回结果最多数量
            limit=3
        )

        if filter_result["success"]:
            print(f"✅ 过滤搜索成功")
            print(f"   结果数量: {filter_result['total_results']}")

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
                        f"相似度得分: {distance:.4f}\n"
                        f"{'-' * 50}\n"
                    )
                    filtered_result_string += record
                print(f"过滤搜索结果:\n{filtered_result_string}")
        else:
            print(f"❌ 过滤搜索失败: {filter_result['error']}")
            if "suggestions" in filter_result:
                print(f"   建议查询: {filter_result['suggestions']}")


    except Exception as e:
        logger.error(f"主程序执行异常: {e}")
        print(f"❌ 程序异常终止: {e}")
