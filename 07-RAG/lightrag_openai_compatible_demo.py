import os
import asyncio
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed, openai_complete_if_cache_vllm
from lightrag.llm.ollama import xinference_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache_vllm(
        "QwQ-32B",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="EMPTY",
        base_url="http://172.17.136.62:10081/v1",
        **kwargs,
    )


async def embedding_func_bak(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="BCE-embedding-base_v1",
        api_key="EMPTY",
        base_url="http://172.17.136.62:9997/v1",
    )
    
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await xinference_embedding(
        texts,
        model="BCE-embedding-base_v1",
        base_url="http://172.17.136.62:9997",
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("介绍下自己，想象下未来你的生活?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


# asyncio.run(test_funcs())


async def initialize_rag():
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        entity_extract_max_gleaning=0,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
        addon_params={
            "example_number": 1,
            "language": "Simplfied Chinese",
        }
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()

        with open("./dickens/深圳市气象灾害应急预案.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # Perform naive search
        print(
            await rag.aquery(
                "市气象灾害应急指挥部的成员单位有哪些?", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        print(
            await rag.aquery(
                "市气象灾害应急指挥部的成员单位有哪些?", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print(
            await rag.aquery(
                "市气象灾害应急指挥部的成员单位有哪些?",
                param=QueryParam(mode="global"),
            )
        )

        # Perform hybrid search
        print(
            await rag.aquery(
                "市气象灾害应急指挥部的成员单位有哪些?",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
