
# 08-Agent

## 目录


- [文件目录说明](#文件目录说明)
- [版本控制](#版本控制)
- [作者](#作者)
- [版权说明](#版权说明)
- [鸣谢](#鸣谢)


### 文件目录说明
eg:

```
├─0801-LangGraph_demo
│  │  01.png
│  │  apiTest.py
│  │  demoRagAgent.py
│  │  docker-compose.yml
│  │  main.py
│  │  README.MD
│  │  vectorSaveTest.py
│  │  webUI.py
│  ├─input
│  │      健康档案.pdf
│  ├─output
│  │      app.log
│  ├─prompts
│  │      prompt_template_agent.txt
│  │      prompt_template_generate.txt
│  │      prompt_template_grade.txt
│  │      prompt_template_rewrite.txt
│  └─utils
│      │  config.py
│      │  llms.py
│      │  pdfSplitTest_Ch.py
│      │  pdfSplitTest_En.py
│      │  tools_config.py
│      └─__pycache__
│              config.cpython-311.pyc
│              llms.cpython-311.pyc
│              pdfSplitTest_Ch.cpython-311.pyc
│              pdfSplitTest_En.cpython-311.pyc
│              tools_config.cpython-311.pyc
├─0802-ReActAgent_demo
│  │  01_backendServer.py
│  │  02_frontendServer.py
│  │  README.md
│  ├─docker
│  │  ├─postgresql
│  │  │      docker-compose.yml
│  │  └─redis
│  │          docker-compose.yaml
│  ├─docs
│  │      01_后端业务核心流程.pdf
│  │      02_API接口和数据模型描述.pdf
│  │      03_前端业务核心流程.pdf
│  └─utils
│          config.py
│          llms.py
│          tools.py
└─0803-RagwithMilvus_demo
        .env
        00-RSS公众号文章自动同步工作流.json
        01_createDatabase.py
        02_createCollection.py
        03_insertData.py
        04_basicSearch.py
        05_fullTextSearch.py
        06_hybridTextSearch.py
        clientChatTest.py
        docker-compose.yml
        milvusSearchMCPServer.py
        milvusSearchMCPServerTest.py
        mixTextSearch.py
        README.md
        servers_config.json
        streamableHttpStart.py
        test.json
```

#### 0801-LangGraph_demo

本项目实现了一个基于状态图的对话流程，通过分析用户输入、调用工具和生成回复，提供流畅的交互体验。以下是其核心工作流程：

1. 用户输入问题后，进入 **agent 分诊节点**，进行意图分析。
2. 若需调用工具，则路由到 **call_tools 节点**，并行执行工具调用；否则直接生成回复并结束流程。
3. 根据工具类型：
   - 若为检索类工具，进入 **grade_documents 节点** 进行相关性评分。若评分相关，则路由到 **generate 节点** 生成回复；否则最多重写（rewrite）3次。
   - 若为非检索类工具，直接路由到 **generate 节点** 生成回复。
4. 最终回复生成后，输出至用户。

#### 0802-ReActAgent_demo

本项目基于FastAPI框架构建了一个功能完整的Agent智能体API后端服务，集成了LangGraph ReAct架构、多类型记忆管理、工具调用和人工审查等核心功能，为AI应用提供强大的智能体服务能力。

#### 0803-RagwithMilvus_demo

本项目将Milvus向量数据库封装为MCP（Model Context Protocol）Server，为AI应用提供高效的知识库搜索服务。通过标准化的MCP协议，大语言模型可以方便地调用Milvus的向量搜索能力，实现智能问答和知识检索。

### 贡献者

GuoFeng


### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。

### 作者

[GuoFeng Github](https://github.com/hquzhuguofeng)

[GuoFeng CSDN](https://blog.csdn.net/weixin_46133588?spm=1011.2415.3001.5343)

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*



### 鸣谢
- [NanGeAGI](https://space.bilibili.com/509246474/upload/video)