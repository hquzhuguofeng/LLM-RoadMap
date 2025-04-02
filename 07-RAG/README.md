



# LightRAG 快速入门指南

## 安装 LightRAG 核心

### 推荐安装方式：从源代码安装

```bash
cd LightRAG
pip install -e .
```

### 从 PyPI 安装

```bash
pip install lightrag-hku
```

## 后端服务部署

### 大模型部署（使用 VLLM）

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/pretrain_model/Qwen/QwQ-32B \
  --served-model-name QwQ-32B \
  --gpu-memory-utilization 0.85 \
  --port 1891 \
  --max-model-len 12000
```

### Embedding 模型部署（使用 Xinference）

配置文件示例：
```json
{
    "model_name": "BCE-embedding-base_v1",
    "model_id": null,
    "model_revision": null,
    "model_hub": "huggingface",
    "dimensions": 768,
    "max_tokens": 512,
    "language": ["en", "zh"],
    "model_uri": "/workspace/pretrain_model/embedding_models/bce-embedding-base_v1",
    "is_builtin": false
}
```

## 前端代码

LightRAG 前端代码位于以下文件中：
```bash
07-RAG/lightrag_openai_compatible_demo.py
```

---

### 📝 注意事项
- 确保在部署前已安装所有依赖项。
- 根据实际硬件配置调整 `--gpu-memory-utilization` 和 `--max-model-len` 参数。
- 配置文件中的 `model_uri` 需根据实际路径修改。

