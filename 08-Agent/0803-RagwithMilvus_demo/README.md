# 1、封装milvus数据库搜索服务为MCP Server对外提供使用

MCP官方简介:https://www.anthropic.com/news/model-context-protocol                                                                                           
MCP文档手册:https://modelcontextprotocol.io/introduction                                                        
MCP官方服务器列表:https://github.com/modelcontextprotocol/servers                                  
PythonSDK的github地址:https://github.com/modelcontextprotocol/python-sdk    

## 1.1 milvus介绍  
Milvus 是一个开源云原生向量数据库，专为在海量向量数据集上进行高性能相似性搜索而设计       
它建立在流行的向量搜索库（包括 Faiss、HNSW、DiskANN 和 SCANN）之上，可为人工智能应用和非结构化数据检索场景提供支持          
官方地址:https://milvus.io/docs/zh    

## 1.2 使用在线n8n平台爬取文章           
新建工作流，使用文件夹中提供的 RSS公众号文章自动同步工作流.json 文件，并对该工作流节点进行配置修改                                                              
公众号文章RSS来源使用国内领先、稳定、快速的一站式公众号RSS订阅平台，专注于精品内容的开放                                              
注册链接:http://www.jintiankansha.me/account/signup?invite_code=PJIFQVLQKJ                   
json格式化工具:https://tool.browser.qq.com/jsonbeautify.html    

## 1.3 部署milvus数据库服务                  
参考官方文档进行安装部署 https://milvus.io/docs/zh/install_standalone-docker-compose.md           
**安装Docker:**           
官网链接如下:https://www.docker.com/ 根据自己的操作系统选择对应的Desktop版本下载安装                                    
安装成功之后启动Docker Desktop软件即可                 
**运行指令:**                  
打开命令行终端，进入到docker-compose.yml所在文件夹，运行以下指令:                           
docker compose up -d             

## 1.4 基于milvus进行知识库搭建 
### 1.4.1 使用milvus
milvus数据库服务启动成功之后，安装官方提供的可视化客户端进行使用，参考如下说明进行安装部署                              
https://github.com/zilliztech/attu               
直接使用docker的方式进行部署使用也可以下载打包好的release包直接使用                  
http://localhost:8000  

### 1.4.2 创建数据库      
打开命令行终端，运行 `python 01_createDatabase.py` 脚本          

### 1.4.3 创建集合    
打开命令行终端，运行 `python 02_createCollection.py` 脚本                  
根据要存储的业务数据进行schema定义、创建索引、集合创建，业务数据如下所示:             
{           
    "docId": "article_b7f27241-1c45-4422-8032-0ee0851e4883",             
    "title": "AI智能体是否能预测未来？字节跳动seed发布FutureX动态评测基准",                
    "link": "http://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650988457&idx=3&sn=f7e1569141a6aaa26e20a9121c9a",             
    "pubDate": "2025.08.31 13:30:00",            
    "pubAuthor": "机器之心",              
    "content": "你有没有想过，AI 不仅能记住过去的一切，还能预见未知的未来？\n\n想象一下，让 AI 预测下周的股价……                
}            

### 1.4.5 插入数据 
打开命令行终端，运行 `python 03_insertData.py` 脚本  

### 1.4.6 数据搜索
(1)语义搜索和查询测试                       
打开命令行终端，运行 `python 04_basicSearch.py` 脚本                       
(2)全文搜索测试(关键词全文匹配)                        
打开命令行终端，运行 `python 05_fullTextSearch.py` 脚本                       
(3)混合搜索测试(语义搜索+全文搜索)                        
打开命令行终端，运行 `python 06_hybridTextSearch.py` 脚本   
                    
## 1.5 milvus数据库搜索MCP Server测试
打开命令行终端，首先运行 `python streamableHttpStart.py` 脚本启动MCP Server服务                  
再运行 `python milvusSearchMCPServerTest.py` 脚本测试，进行服务接口的单独验证测试                                
最后运行 `python clientChatTest.py` 脚本测试，使用大模型进行测试，在运行脚本之前，需要在.env文件中配置大模型相关的参数及在servers_config.json文件中配置需要使用的MCP Server      
按照如下参考问题进行测试:       
(1)不指定搜索类型和条件过滤，默认进行混合搜索，默认没有过滤条件             
搜索关于时序增强关系敏感知识迁移相关的文章，并给出文章的标题、链接、发布者                
(2)指定搜索类型和条件过滤    
全文搜索关于AI浏览器公司相关的文章，文章发布时间在2025.09.06之前，返回3篇文章并给出文章的标题、链接、发布者              
语义搜索关于时序增强关系敏感知识迁移相关的文章，文章发布时间在2025.09.17之前，返回3篇文章并给出文章的标题、链接、发布者               
混合搜索关于时序增强关系敏感知识迁移相关的文章，文章发布时间在2025.09.16之前，返回3篇文章并给出文章的标题、链接、发布者               
(3)不指定搜索类型但指定条件过滤(存在多个过滤条件)            
搜索关于时序增强关系敏感知识迁移相关的文章，文章发布时间在2025.09.11之前，发布者是量子位，返回3篇文章并给出文章的标题、链接、发布者            
(4)不指定搜索类型但指定条件过滤(存在无关过滤条件干扰)           
搜索关于时序增强关系敏感知识迁移相关的文章，文章发布时间在2025.09.11之前，发布者是新智元，字数在200字内，价格不超过500元，返回3篇文章并给出文章的标题、链接、发布者           

## 1.6 如何使用MCP服务

```
async def get_tools():
    # 自定义工具 模拟酒店预定工具
    @tool("book_hotel", description="酒店预定工具")
    async def book_hotel(hotel_name: str):
        """
       支持酒店预定的工具

        Args:
            hotel_name: 酒店名称

        Returns:
            工具的调用结果
        """
        return f"成功预定了在{hotel_name}的住宿。"

    # 自定义工具 计算两个数的乘积的工具
    @tool("multiply", description="计算两个数的乘积的工具")
    async def multiply(a: float, b: float) -> float:
        """
       支持计算两个数的乘积的工具

        Args:
            a: 参数1
            b: 参数2

        Returns:
            工具的调用结果
        """
        result = a * b
        return f"{a}乘以{b}等于{result}。"

    # # MCP Server工具 高德地图
    # client = MultiServerMCPClient({
    #     # 高德地图MCP Server
    #     "amap-amap-sse": {
    #         "url": "https://mcp.amap.com/sse?key=848232bewe1987634de9ew23e19wewed61265e50bb0757",
    #         "transport": "sse",
    #     }
    # })
    # # 从MCP Server中获取可提供使用的全部工具
    # amap_tools = await client.get_tools()
    # # 为工具添加人工审查
    # tools = [await add_human_in_the_loop(index) for index in amap_tools]
    #
    # # 追加自定义工具并添加人工审查
    # tools.append(await add_human_in_the_loop(book_hotel))
    # tools.append(multiply)
    #
    # # 返回工具列表
    # return tools

    # MCP Server工具 基于milvus的搜索
    client = MultiServerMCPClient({
        "rag_mcp_server": {
            "url": "http://127.0.0.1:8010/mcp",
            "transport": "streamable_http",
        }
    })
    # 从MCP Server中获取可提供使用的全部工具
    tools = await client.get_tools()
    # # 为工具添加人工审查
    tools = [await add_human_in_the_loop(index) for index in tools]

    # 追加自定义工具并添加人工审查
    tools.append(await add_human_in_the_loop(book_hotel))
    tools.append(multiply)

    # 返回工具列表
    return tools
```