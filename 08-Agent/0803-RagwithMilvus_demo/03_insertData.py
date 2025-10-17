from pymilvus import MilvusClient, DataType, Function, FunctionType
from openai import OpenAI
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Callable
import logging
import time
import random
import json



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 配置日志
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MilvusDataInserter:
	"""Milvus数据插入管理器，提供文档分块插入功能"""

	def __init__(self,
				 milvus_uri: str = "http://172.17.136.62:19530",
				 db_name: str = "my_milvus_database_demo",
				 openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
				 openai_api_key: str = "sk-25be5a383bd8450eb701bd44da3a8cd2"):
		"""
		初始化数据插入管理器

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

			# 初始化Milvus客户端
			self.milvus_client = MilvusClient(
				milvus_uri=self.milvus_uri,
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
					dimensions=1536,
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

	def split_text_into_chunks(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
		"""
		将文本分割成指定大小的块

		Args:
			text: 要分割的文本
			chunk_size: 每块的字符数，默认800
			overlap: 块之间的重叠字符数，默认100

		Returns:
			List[str]: 分割后的文本块列表
		"""
		try:
			# 参数验证
			if not text or not isinstance(text, str):
				logger.warning("输入文本为空或非字符串类型")
				return []

			if chunk_size <= 0:
				raise ValueError("chunk_size必须大于0")

			if overlap < 0:
				raise ValueError("overlap不能为负数")

			if overlap >= chunk_size:
				logger.warning("overlap大于等于chunk_size，调整overlap为chunk_size的一半")
				overlap = chunk_size // 2

			# 如果文本长度小于等于chunk_size，直接返回
			if len(text) <= chunk_size:
				return [text]

			chunks = []
			start = 0

			while start < len(text):
				end = start + chunk_size

				# 如果不是最后一块，尝试在句号、感叹号、问号处断句
				if end < len(text):
					# 在当前块的后100个字符范围内寻找合适的断句点
					search_end = min(end + 100, len(text))
					sentence_end = max(
						text.rfind('。', end - 100, search_end),
						text.rfind('！', end - 100, search_end),
						text.rfind('？', end - 100, search_end),
						text.rfind('\n', end - 100, search_end)
					)

					if sentence_end > end - 100:
						end = sentence_end + 1

				chunk = text[start:end].strip()
				if chunk:
					chunks.append(chunk)

				# 设置下一块的起始位置，考虑重叠
				start = max(start + chunk_size - overlap, end)

				if start >= len(text):
					break

			logger.debug(f"文本分割完成，共生成 {len(chunks)} 个块")
			return chunks

		except Exception as e:
			logger.error(f"文本分割失败: {e}")
			return [text] if text else []

	def validate_document(self, doc_data: Dict[str, Any], doc_idx: int) -> bool:
		"""
		验证文档数据的完整性

		Args:
			doc_data: 文档数据字典
			doc_idx: 文档索引

		Returns:
			bool: 验证是否通过
		"""
		try:
			# 验证必需字段
			required_fields = ['docId', 'title', 'content', 'link', 'pubDate', 'pubAuthor']

			for field in required_fields:
				if field not in doc_data:
					logger.error(f"文档 {doc_idx} 缺少必需字段: {field}")
					return False

				if not doc_data[field]:
					logger.warning(f"文档 {doc_idx} 的字段 {field} 为空")

			# 验证字段类型和长度
			if not isinstance(doc_data['docId'], str) or len(doc_data['docId']) > 100:
				logger.error(f"文档 {doc_idx} 的docId无效")
				return False

			if not isinstance(doc_data['title'], str) or len(doc_data['title']) > 1000:
				logger.error(f"文档 {doc_idx} 的title无效")
				return False

			if not isinstance(doc_data['content'], str):
				logger.error(f"文档 {doc_idx} 的content无效")
				return False

			return True

		except Exception as e:
			logger.error(f"验证文档 {doc_idx} 时出错: {e}")
			return False

	def batch_insert_documents_with_chunks(self,
										   collection_name: str,
										   documents: List[Dict[str, Any]],
										   chunk_size: int = 800,
										   overlap: int = 100,
										   batch_size: int = 10) -> Dict[str, Any]:
		"""
		批量将文档分块后插入到集合中

		Args:
			collection_name: 集合名称
			documents: 文档数据列表
			chunk_size: 每块的字符数，默认800
			overlap: 块之间的重叠字符数，默认100
			batch_size: 批量插入的大小，默认10

		Returns:
			Dict: 插入结果统计信息
		"""
		try:
			# 参数验证
			if not collection_name or not isinstance(collection_name, str):
				raise ValueError("集合名称不能为空且必须是字符串类型")

			if not documents or not isinstance(documents, list):
				logger.warning("没有文档需要插入")
				return {"total_documents": 0, "total_chunks": 0, "success": True}

			if batch_size <= 0:
				raise ValueError("batch_size必须大于0")

			# 检查集合是否存在
			if not self.milvus_client.has_collection(collection_name):
				raise ValueError(f"集合 '{collection_name}' 不存在")

			all_chunks = []
			document_stats = {}
			failed_documents = []

			logger.info(f"开始处理 {len(documents)} 个文档...")

			# 使用进度条处理每个文档
			for doc_idx, doc_data in enumerate(tqdm(documents, desc="处理文档")):
				try:
					# 验证文档数据
					if not self.validate_document(doc_data, doc_idx):
						failed_documents.append(doc_idx)
						continue

					# 分割内容
					content_chunks = self.split_text_into_chunks(
						doc_data['content'],
						chunk_size=chunk_size,
						overlap=overlap
					)

					if not content_chunks:
						logger.warning(f"文档 {doc_data['docId']} 分割后无有效块")
						failed_documents.append(doc_idx)
						continue

					document_stats[doc_data['docId']] = len(content_chunks)

					# 为每个块创建数据
					for chunk_idx, chunk in enumerate(content_chunks):
						try:
							# 生成密集向量嵌入
							dense_vector = self.emb_text(chunk)

							chunk_data = {
								"docId": str(doc_data['docId']),
								"chunk_index": chunk_idx,
								"title": str(doc_data['title'])[:1000],  # 确保不超过长度限制
								"link": str(doc_data['link'])[:500],
								"pubDate": str(doc_data['pubDate'])[:100],
								"pubAuthor": str(doc_data['pubAuthor'])[:100],
								"content_chunk": chunk[:3000],  # 确保不超过长度限制
								"content_dense": dense_vector,
								"full_content": str(doc_data['content'])[:20000]
							}
							all_chunks.append(chunk_data)

						except Exception as e:
							logger.error(f"处理文档 {doc_data['docId']} 的块 {chunk_idx} 时出错: {e}")
							continue

					logger.debug(f"文档 {doc_data['docId']} 已分割为 {len(content_chunks)} 个块")

				except Exception as e:
					logger.error(f"处理文档 {doc_idx} 时出错: {e}")
					failed_documents.append(doc_idx)
					continue

			if not all_chunks:
				logger.error("没有有效的文档块需要插入")
				return {
					"total_documents": len(documents),
					"total_chunks": 0,
					"inserted_chunks": 0,
					"failed_batches": 0,
					"failed_documents": failed_documents,
					"document_stats": document_stats,
					"success": False
				}

			# 批量插入数据
			logger.info(f"开始批量插入 {len(all_chunks)} 个文档块...")

			total_inserted = 0
			failed_batches = []

			# 使用进度条分批插入
			for i in tqdm(range(0, len(all_chunks), batch_size), desc="插入数据"):
				batch_data = all_chunks[i:i + batch_size]
				batch__data_dicts = [
					{
						"docId": chunk["docId"],
						"chunk_index": chunk["chunk_index"],
						"title": chunk["title"],
						"link": chunk["link"],
						"pubDate": chunk["pubDate"],
						"pubAuthor": chunk["pubAuthor"],
						"content_chunk": chunk["content_chunk"],
						"content_dense": chunk["content_dense"],
						"full_content": chunk["full_content"]
					} for chunk in batch_data
				]
				try:
					result = self.milvus_client.insert(collection_name=collection_name, data=batch_data)
					total_inserted += len(batch_data)
					logger.info(f"批次 {(i // batch_size) + 1}: 成功插入 {len(batch_data)} 个块")
				except Exception as e:
					logger.error(f"批次 {(i // batch_size) + 1} 插入失败: {e}")
					failed_batches.append((i // batch_size) + 1)
					continue

			logger.info("=== 批量插入完成 ===")
			logger.info(f"处理文档数: {len(documents)}")
			logger.info(f"生成文档块数: {len(all_chunks)}")
			logger.info(f"成功插入块数: {total_inserted}")
			logger.info(f"失败批次数: {len(failed_batches)}")

			logger.info("=== 各文档分块统计 ===")
			for docId, chunk_count in document_stats.items():
				logger.info(f"文档 {docId}: {chunk_count} 个块")

			return {
				"total_documents": len(documents),
				"total_chunks": len(all_chunks),
				"inserted_chunks": total_inserted,
				"failed_batches": len(failed_batches),
				"failed_documents": failed_documents,
				"document_stats": document_stats,
				"success": len(failed_batches) == 0
			}

		except Exception as e:
			logger.error(f"批量插入失败: {e}")
			return {
				"total_documents": len(documents),
				"total_chunks": 0,
				"inserted_chunks": 0,
				"failed_batches": 0,
				"failed_documents": [],
				"document_stats": {},
				"success": False
			}


if __name__ == "__main__":
	# 实例化插入管理器
	inserter = MilvusDataInserter(
		milvus_uri="http://localhost:19530",
		db_name="milvus_database",
		openai_base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1",
		openai_api_key= "sk-25be5a383bd8450eb701bd44da3a8cd2"
	)

	# 读取本地json文件
	def read_json_file(file_path):
		try:
			with open(file_path, 'r', encoding='utf-8') as file:
				data = json.load(file)
				return data
		except FileNotFoundError:
			logger.error(f"错误：文件 {file_path} 未找到。")
			return None
		except json.JSONDecodeError:
			logger.error(f"错误：文件 {file_path} 包含无效的 JSON 格式。")
			return None
		except Exception as e:
			logger.error(f"错误：发生意外错误：{str(e)}")
			return None

	# 指定文件路径
	file_path = "./test.json"
	# 读取并打印 JSON 内容
	json_data = read_json_file(file_path)
	if json_data is not None:
		logger.info(json_data)
		# 执行批量插入
		insert_result = inserter.batch_insert_documents_with_chunks(
			collection_name="my_collection_demo_chunked",
			documents=json_data,
			chunk_size=800,
			overlap=100,
			batch_size=10
		)
		logger.info(f"插入结果: {insert_result}")
