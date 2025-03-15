# 🌟 领域自动问答对生成项目

> 本项目旨在通过结合文档加载、文本清理、异步LLM API调用和问答生成等技术，实现对特定领域文档的自动问答对生成。✨ 项目亮点如下：
 - 💻 后端基于 **vllm** 部署 **deepseek-r1-32b**
 - 🚀 客户端使用异步方式，高效调用后端 LLM
 - 🔍 核心内容：读取文档、切片处理，构建问题，并基于 **RAG（Retrieval-Augmented Generation）** 实现高质量问答对生成
 - 📊 实现垂直领域问答对的高效构建，助力企业知识库建设与智能问答系统开发

---

## 📂 项目结构

```bash
├── ./data  # 数据内容示例
│   ├── ./中华人民共和国民法典-2021.docx
├── ./async_llm_api_batch_data.py  # 异步调用 LLM 类实现
├── ./mydocloader                  # 文档加载模块
├── ./prompt                       # 生成问题的 Prompt 模板
├── ./query_gen.py                 # 主程序入口
├── ./retrival_doc.py              # 基于 RAG 的问题回复模块
└── ./README.md                    # 项目说明文档
```

---

## 🛠️ 核心功能

### 1. 文档加载与处理 📄

- ✅ 支持 PDF/图片文档解析（基于 RapidOCR）
- ✅ 文本清洗：去除冗余空格、换行符、特殊字符等
- ✅ 动态分块：支持自定义块大小，灵活适配不同文档长度

### 2. 异步问答生成 ⚡

- ✅ 基于 **LangChain** 的异步 API 调用，提升效率
- ✅ 可配置的 QPS（Queries Per Second）限制，确保稳定运行

#### 2.1 `asyncio.Semaphore` 🔒
适用于需要严格控制并发数的场景：
- 数据库连接池管理
- 多线程爬虫抓取网页
- 防止 CPU 或内存资源被过度占用

#### 2.2 `limiter` 🔄
适用于需要平滑流量的场景：
- 调用带有速率限制的第三方 API
- 分布式系统的负载均衡
- 高频交易或其他对速率敏感的应用

### 3. 检索增强生成 🧠

- ✅ 结合文档片段检索生成答案，确保上下文相关性
- ✅ 输出格式化为 JSONL，便于后续处理
- ✅ 自动生成上下文关联的问答对，提升问答质量

---

## 🚀 运行步骤

### a. API 设置 🌐

在代码中配置以下参数：

```python
llm = CustomAsyncLLMAPI(
    base_url='http://your-api-endpoint/v1',  # 替换实际 API 地址
    api_key='your-api-key',
    model='your-model-name',
    max_concurrency=100,  # 设置最大并发数
    max_rate=100          # 设置每秒最大请求数
)
```

### b. 文档处理参数 📝

调整以下参数以满足需求：

```python
# 在 CustomAsyncLLMAPI.custom_async_run 中调整：
chunk_size=2000  # 文档分块大小（字符数）
output_name='自定义输出文件名'  # 输出到 output 目录
```

### c. 启动程序 🏃‍♂️

运行以下命令启动程序：

🍒🍒🍒 `python query_gen.py` 🍒🍒🍒

---

## ⚠️ 注意事项

- **Windows 系统需设置事件循环策略**：
```python
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

- **依赖安装**：确保已安装所有依赖项，可通过 `requirements.txt` 安装：
```bash
pip install -r requirements.txt
```

- **API 密钥安全**：请妥善保管您的 API 密钥，避免泄露。

---

## 🌈 总结

本项目通过高效的异步调用和强大的 RAG 技术，实现了垂直领域文档的自动问答对生成，助力企业快速构建智能化的知识问答系统。🌟 如果您有任何问题或建议，欢迎提交 Issue 或 PR！🎉