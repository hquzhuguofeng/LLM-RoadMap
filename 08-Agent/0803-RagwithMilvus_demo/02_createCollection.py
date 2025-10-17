from pymilvus import MilvusClient, DataType, Function, FunctionType
import logging
from typing import Optional, Dict, Any, List
import time



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MilvusCollectionManager:
    """Milvus集合管理器，提供集合创建、管理等功能"""

    def __init__(self, uri: str = "http://localhost:19530", db_name: str = "my_milvus_database_demo"):
        """
        初始化Milvus集合管理器

        Args:
            uri: Milvus服务器地址
            db_name: 数据库名称
        """
        self.uri = uri
        self.db_name = db_name
        self.client = None

    def connect(self, timeout: float = 30.0) -> bool:
        """
        连接到Milvus服务器

        Args:
            timeout: 连接超时时间（秒）

        Returns:
            bool: 连接是否成功
        """
        try:
            # 验证连接参数
            if not self.uri or not isinstance(self.uri, str):
                raise ValueError("URI不能为空且必须是字符串类型")

            if not self.db_name or not isinstance(self.db_name, str):
                raise ValueError("数据库名称不能为空且必须是字符串类型")

            logger.info(f"正在连接到Milvus服务器: {self.uri}, 数据库: {self.db_name}")

            # 实例化Milvus客户端对象
            self.client = MilvusClient(
                uri=self.uri,
                db_name=self.db_name,
                timeout=timeout
            )

            # 测试连接是否成功
            try:
                collections = self.client.list_collections()
                logger.info(f"成功连接到Milvus服务器，当前集合数量: {len(collections)}")
                return True
            except Exception as e:
                raise ConnectionError(f"无法连接到Milvus服务器或数据库不存在: {e}")

        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False

    def create_schema(self) -> Optional[Any]:
        """
        创建集合的Schema定义

        Returns:
            Optional[Any]: Schema对象，失败时返回None
        """
        try:
            logger.info("开始创建Schema定义...")

            # 定义schema，启用动态字段支持
            schema = MilvusClient.create_schema(enable_dynamic_field=True)

            # === 字段定义 ===
            logger.info("添加字段定义...")

            # 主键字段：文档块ID（自动生成）
            schema.add_field(
                field_name="id",
                datatype=DataType.INT64,
                auto_id=True,
                is_primary=True,
                description="文档块id"
            )

            # 原始文档信息字段
            schema.add_field(
                field_name="docId",
                datatype=DataType.VARCHAR,
                max_length=100,
                description="原始文档唯一标识"
            )
            schema.add_field(
                field_name="chunk_index",
                datatype=DataType.INT64,
                description="文档块在原文档中的序号"
            )

            # 文章标题字段 支持稀疏向量全文搜索
            analyzer_params = {"type": "chinese"}
            schema.add_field(
                field_name="title",
                datatype=DataType.VARCHAR,
                max_length=1000,
                analyzer_params=analyzer_params,
                enable_match=True,
                enable_analyzer=True,
                description="文章标题"
            )
            schema.add_field(
                field_name="title_sparse",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
                description="文章标题的稀疏嵌入由内置BM25函数自动生成"
            )

            # 其他元数据字段
            schema.add_field(
                field_name="link",
                datatype=DataType.VARCHAR,
                max_length=500,
                description="文章原始链接地址"
            )
            schema.add_field(
                field_name="pubDate",
                datatype=DataType.VARCHAR,
                max_length=100,
                description="发布时间"
            )
            schema.add_field(
                field_name="pubAuthor",
                datatype=DataType.VARCHAR,
                max_length=100,
                description="发布者"
            )
            schema.add_field(
                field_name="full_content",
                datatype=DataType.VARCHAR,
                max_length=50000,
                description="原始完整文章内容"
            )

            # 文档块内容字段 支持密集向量语义搜索和稀疏向量关键词搜索
            schema.add_field(
                field_name="content_chunk",
                datatype=DataType.VARCHAR,
                max_length=3000,
                analyzer_params=analyzer_params,
                enable_match=True,
                enable_analyzer=True,
                description="文档内容块(最大800字符)"
            )
            schema.add_field(
                field_name="content_dense",
                datatype=DataType.FLOAT_VECTOR,
                dim=1536,
                description="文档块的密集向量嵌入"
            )
            schema.add_field(
                field_name="content_sparse",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
                description="文档块的稀疏向量嵌入"
            )

            logger.info("字段定义添加完成")
            return schema

        except Exception as e:
            logger.error(f"创建Schema失败: {e}")
            return None

    def add_bm25_functions(self, schema: Any) -> bool:
        """
        为Schema添加BM25函数

        Args:
            schema: Schema对象

        Returns:
            bool: 操作是否成功
        """
        try:
            logger.info("添加BM25函数定义...")

            # 定义标题BM25函数
            title_bm25_function = Function(
                name="title_bm25_emb",
                input_field_names=["title"],
                output_field_names=["title_sparse"],
                function_type=FunctionType.BM25,
            )

            # 定义内容BM25函数
            content_bm25_function = Function(
                name="content_bm25_emb",
                input_field_names=["content_chunk"],
                output_field_names=["content_sparse"],
                function_type=FunctionType.BM25,
            )

            # 将函数添加到schema
            schema.add_function(title_bm25_function)
            schema.add_function(content_bm25_function)

            logger.info("BM25函数添加完成")
            return True

        except Exception as e:
            logger.error(f"添加BM25函数失败: {e}")
            return False

    def create_index_params(self) -> Optional[Any]:
        """
        创建索引参数配置

        Returns:
            Optional[Any]: 索引参数对象，失败时返回None
        """
        try:
            logger.info("开始创建索引参数...")

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()

            # 主键索引
            index_params.add_index(
                field_name="id",
                index_type="AUTOINDEX"
            )
            logger.info("添加主键索引")

            # 文档ID索引，便于查询同一文档的所有块
            index_params.add_index(
                field_name="docId",
                index_type="AUTOINDEX"
            )
            logger.info("添加文档ID索引")

            # 稀疏向量索引 - 标题
            index_params.add_index(
                field_name="title_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",  # 算法选择
                    "bm25_k1": 1.2,  # 词频饱和度控制参数
                    "bm25_b": 0.75  # 文档长度归一化参数
                }
            )
            logger.info("添加标题稀疏向量索引")

            # 稀疏向量索引 - 内容
            index_params.add_index(
                field_name="content_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    "bm25_k1": 1.2,
                    "bm25_b": 0.75
                }
            )
            logger.info("添加内容稀疏向量索引")

            # 密集向量索引
            index_params.add_index(
                field_name="content_dense",
                index_type="AUTOINDEX",
                metric_type="COSINE"  # 使用余弦相似度
            )
            logger.info("添加密集向量索引")

            logger.info("索引参数创建完成")
            return index_params

        except Exception as e:
            logger.error(f"创建索引参数失败: {e}")
            return None

    def create_collection(
            self,
            collection_name: str = "my_collection_demo_chunked",
            drop_existing: bool = True,
            wait_for_load: bool = True,
            load_timeout: int = 60
    ) -> bool:
        """
        创建集合的完整流程

        Args:
            collection_name: 集合名称
            drop_existing: 是否删除已存在的同名集合
            wait_for_load: 是否等待集合加载完成
            load_timeout: 加载超时时间（秒）

        Returns:
            bool: 操作是否成功
        """
        try:
            logger.info(f"开始创建集合: {collection_name}")

            # 验证集合名称
            if not collection_name or not isinstance(collection_name, str):
                raise ValueError("集合名称不能为空且必须是字符串类型")

            # 集合名称格式验证
            if not collection_name.replace('_', '').replace('-', '').isalnum():
                raise ValueError("集合名称只能包含字母、数字、下划线和连字符")

            # 检查集合是否已存在
            if self.client.has_collection(collection_name):
                if drop_existing:
                    logger.warning(f"集合 '{collection_name}' 已存在，正在删除...")
                    self.client.drop_collection(collection_name)
                    logger.info(f"集合 '{collection_name}' 删除成功")
                    # 等待一段时间确保删除完成
                    time.sleep(2)
                else:
                    logger.warning(f"集合 '{collection_name}' 已存在，跳过创建")
                    return True

            # 创建Schema
            schema = self.create_schema()
            if schema is None:
                raise RuntimeError("Schema创建失败")

            # 添加BM25函数
            if not self.add_bm25_functions(schema):
                raise RuntimeError("BM25函数添加失败")

            # 创建索引参数
            index_params = self.create_index_params()
            if index_params is None:
                raise RuntimeError("索引参数创建失败")

            # 创建集合
            logger.info(f"正在创建集合 '{collection_name}'...")
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"集合 '{collection_name}' 创建成功")

            # 等待集合加载完成
            if wait_for_load:
                logger.info("等待集合加载完成...")
                start_time = time.time()

                while time.time() - start_time < load_timeout:
                    try:
                        load_state = self.client.get_load_state(collection_name=collection_name)
                        # logger.info(f"集合当前状态:{load_state['state'].name}")
                        if load_state['state'].name == 'Loaded':
                            logger.info("集合加载完成")
                            break
                        elif load_state['state'].name == 'Loading':
                            logger.info("集合正在加载中...")
                            time.sleep(2)
                        else:
                            logger.warning(f"集合加载状态异常: {load_state}")
                            time.sleep(2)
                    except Exception as e:
                        logger.warning(f"获取加载状态时出错: {e}")
                        time.sleep(2)
                else:
                    logger.warning(f"集合加载超时（{load_timeout}秒）")

            return True

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        获取集合的详细信息

        Args:
            collection_name: 集合名称

        Returns:
            Optional[Dict[str, Any]]: 集合信息，失败时返回None
        """
        try:
            logger.info(f"获取集合 '{collection_name}' 的信息...")

            # 检查集合是否存在
            if not self.client.has_collection(collection_name):
                logger.error(f"集合 '{collection_name}' 不存在")
                return None

            # 获取集合状态
            load_state = self.client.get_load_state(collection_name=collection_name)

            # 获取集合详细信息
            collection_info = self.client.describe_collection(collection_name=collection_name)

            # 获取集合统计信息
            try:
                stats = self.client.get_collection_stats(collection_name=collection_name)
            except Exception as e:
                logger.warning(f"获取集合统计信息失败: {e}")
                stats = None

            info = {
                "load_state": load_state,
                "collection_info": collection_info,
                "stats": stats
            }
            logger.info(f"集合信息获取成功: {info}")
            return info
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return None


if __name__ == "__main__":
    manager = MilvusCollectionManager(uri="http://localhost:19530", db_name="milvus_database")
    if manager.connect():
        collection_name = "my_collection_demo_chunked"
        if manager.create_collection(collection_name=collection_name):
            info = manager.get_collection_info(collection_name=collection_name)
            if info:
                print(f"{collection_name}集合的当前状态: {info['load_state']}")
                print(f"{collection_name}集合的详细信息: {info['collection_info']}")
                if info['stats']:
                    print(f"{collection_name}集合的统计信息: {info['stats']}")
        else:
            logger.error("集合创建失败")
    else:
        logger.error("连接Milvus服务器失败")
