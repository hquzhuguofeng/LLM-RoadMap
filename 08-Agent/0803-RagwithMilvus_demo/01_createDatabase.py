from pymilvus import MilvusClient
import logging
from typing import Optional, List



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_milvus_database(
        uri: str = "http://localhost:19530",
        db_name: str = "my_milvus_database_demo",
        timeout: float = 30.0
) -> bool:
    """
    创建Milvus数据库的完整流程

    Args:
        uri: Milvus服务器地址
        db_name: 要创建的数据库名称
        timeout: 连接超时时间（秒）

    Returns:
        bool: 操作是否成功
    """
    client = None

    try:
        logger.info("开始创建Milvus数据库...")

        # 1、实例化Milvus客户端对象
        # 验证URI格式
        if not uri or not isinstance(uri, str):
            raise ValueError("URI不能为空且必须是字符串类型")

        # 验证数据库名称
        if not db_name or not isinstance(db_name, str):
            raise ValueError("数据库名称不能为空且必须是字符串类型")

        # 数据库名称格式验证（只允许字母、数字、下划线，且不能以数字开头）
        if not db_name.replace('_', '').replace('-', '').isalnum():
            raise ValueError("数据库名称只能包含字母、数字、下划线和连字符")

        if db_name[0].isdigit():
            raise ValueError("数据库名称不能以数字开头")

        logger.info(f"正在连接到Milvus服务器: {uri}")

        # 创建客户端连接，添加超时设置
        client = MilvusClient(
            uri=uri,
            timeout=timeout
        )

        # 测试连接是否成功
        try:
            # 通过列出数据库来测试连接
            client.list_databases()
            logger.info("成功连接到Milvus服务器")
        except Exception as e:
            raise ConnectionError(f"无法连接到Milvus服务器: {e}")

        # 2、检查数据库是否已存在
        logger.info("检查数据库是否已存在...")
        existing_databases = client.list_databases()

        if db_name in existing_databases:
            logger.warning(f"数据库 '{db_name}' 已存在，跳过创建")
            return True

        # 3、创建数据库
        logger.info(f"正在创建数据库: {db_name}")
        client.create_database(db_name=db_name)
        logger.info(f"数据库 '{db_name}' 创建成功")

        # 4、验证数据库是否创建成功
        logger.info("验证数据库创建结果...")
        updated_databases = client.list_databases()

        if db_name not in updated_databases:
            raise RuntimeError(f"数据库 '{db_name}' 创建失败，未在数据库列表中找到")

        # 5、查询并显示数据库列表
        logger.info("查询当前所有数据库...")
        databases = client.list_databases()
        logger.info(f"当前数据库列表: {databases}")

        # 显示详细信息
        print("=" * 50)
        print("Milvus数据库操作完成")
        print("=" * 50)
        print(f"服务器地址: {uri}")
        print(f"新创建的数据库: {db_name}")
        print(f"当前数据库总数: {len(databases)}")
        print(f"所有数据库: {', '.join(databases)}")
        print("=" * 50)

        return True

    except ConnectionError as e:
        logger.error(f"连接错误: {e}")
        print(f"❌ 连接失败: {e}")
        return False

    except ValueError as e:
        logger.error(f"参数错误: {e}")
        print(f"❌ 参数错误: {e}")
        return False

    except RuntimeError as e:
        logger.error(f"运行时错误: {e}")
        print(f"❌ 运行时错误: {e}")
        return False

    except Exception as e:
        logger.error(f"未知错误: {e}")
        print(f"❌ 发生未知错误: {e}")
        return False

    finally:
        # 清理资源
        if client:
            try:
                # 注意：MilvusClient通常不需要显式关闭，但可以在这里添加清理逻辑
                logger.info("清理客户端连接")
            except Exception as e:
                logger.warning(f"清理资源时出现警告: {e}")


def list_databases_safely(uri: str = "http://localhost:19530") -> Optional[List[str]]:
    """
    安全地列出所有数据库

    Args:
        uri: Milvus服务器地址

    Returns:
        Optional[List[str]]: 数据库列表，失败时返回None
    """
    try:
        logger.info(f"正在连接到Milvus服务器获取数据库列表: {uri}")
        client = MilvusClient(uri=uri)
        databases = client.list_databases()
        logger.info(f"成功获取数据库列表: {databases}")
        return databases
    except Exception as e:
        logger.error(f"获取数据库列表失败: {e}")
        return None


def check_database_exists(uri: str, db_name: str) -> bool:
    """
    检查指定数据库是否存在

    Args:
        uri: Milvus服务器地址
        db_name: 数据库名称

    Returns:
        bool: 数据库是否存在
    """
    try:
        databases = list_databases_safely(uri)
        if databases is None:
            return False
        return db_name in databases
    except Exception as e:
        logger.error(f"检查数据库存在性时出错: {e}")
        return False


def delete_database_safely(uri: str, db_name: str) -> bool:
    """
    安全地删除数据库

    Args:
        uri: Milvus服务器地址
        db_name: 要删除的数据库名称

    Returns:
        bool: 操作是否成功
    """
    try:
        logger.info(f"正在删除数据库: {db_name}")
        client = MilvusClient(uri=uri)

        # 检查数据库是否存在
        if not check_database_exists(uri, db_name):
            logger.warning(f"数据库 '{db_name}' 不存在，无需删除")
            return True

        # 删除数据库
        client.drop_database(db_name=db_name)
        logger.info(f"数据库 '{db_name}' 删除成功")
        return True

    except Exception as e:
        logger.error(f"删除数据库失败: {e}")
        return False


# 主程序执行
if __name__ == "__main__":
    # 配置参数
    MILVUS_URI = "http://localhost:19530"
    DATABASE_NAME = "milvus_database"

    print("开始Milvus数据库操作...")

    # 执行数据库创建操作
    success = create_milvus_database(
        uri=MILVUS_URI,
        db_name=DATABASE_NAME,
        timeout=30.0
    )

    if success:
        print("✅ 数据库操作成功完成!")

        # 额外操作示例
        print("\n--- 额外操作示例 ---")

        # 再次查询数据库列表
        databases = list_databases_safely(MILVUS_URI)
        if databases:
            print(f"当前数据库列表: {databases}")

        # 检查特定数据库是否存在
        exists = check_database_exists(MILVUS_URI, DATABASE_NAME)
        print(f"数据库 '{DATABASE_NAME}' 是否存在: {exists}")

    else:
        print("❌ 数据库操作失败!")

    print("\n程序执行完毕")

