import os
import random
import asyncio
import pandas as pd
from tqdm import tqdm
from typing import List
from dataclasses import dataclass, field
from aiolimiter import AsyncLimiter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

'''
通过异步编程结合令牌池的设计，可以显著提高大模型 API 的调用效率。关键在于：

1、使用 asyncio 管理异步任务。
2、异步协程限速
3、合理分配令牌以实现负载均衡。
4、将多个协程任务交由 asyncio.gather 并发执行。
5、使用AsyncLimiter限制单次调用的并发
6、增加数据切块的事件处理
这一思路可以应用于需要高并发的场景，例如自然语言处理、实时数据处理等，助力开发者构建高效的 AI 应用系统。

使用vllm启动大模型作为模型调用的后端，模型的调用用异步方式是最快的。
model_name_or_path: qwen/Qwen2.5-7B-Instruct
# adapter_name_or_path: ../saves/qwen2.5-7B/ner_epoch5/
template: qwen
finetuning_type: lora
infer_backend: vllm
vllm_enforce_eager: true

# API_PORT=8000 llamafactory-cli api vllm_api.yaml
'''

def generate_arithmetic_expression(num: int):
    """
    生成数学计算的公式和结果
    """
    # 定义操作符和数字范围，除法
    operators = ["+", "-", "*"]
    expression = (
        f"{random.randint(1, 100)} {random.choice(operators)} {random.randint(1, 100)}"
    )
    num -= 1
    for _ in range(num):
        expression = f"{expression} {random.choice(operators)} {random.randint(1, 100)}"
    result = eval(expression)
    expression = expression.replace("*", "x")
    return expression, result


@dataclass
class AsyncLLMAPI:
    """
    大模型API的调用类
    """

    base_url: str
    api_key: str  # 每个API的key不一样
    uid: int = 0
    cnt: int = 0  # 统计每个API被调用了多少次
    model: str = "gpt-3.5-turbo"
    llm: ChatOpenAI = field(init=False)  # 自动创建的对象，不需要用户传入
    num_per_second: int = 6  # 限速每秒调用6次

    def __post_init__(self):
        # 初始化 llm 对象
        self.llm = self.create_llm()
        # 创建限速器，每秒最多发出 5 个请求
        self.limiter = AsyncLimiter(self.num_per_second, 1)

    def create_llm(self):
        # 创建 llm 对象
        return ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

    async def __call__(self, text):
        # 异步协程 限速
        self.cnt += 1
        async with self.limiter:
            return await self.llm.agenerate([text])

    @staticmethod
    async def _run_task_with_progress(task, pbar):
        """包装任务以更新进度条"""
        result = await task
        pbar.update(1)
        return result

    @staticmethod
    def async_run(
        llms: List["AsyncLLMAPI"],
        data: List[str],
        keyword: str = "",  # 文件导出名
        output_dir: str = "output",
        chunk_size=500,
    ):

        async def _func(llms, data):
            """
            异步请求处理一小块数据
            """
            results = [llms[i % len(llms)](text) for i, text in enumerate(data)]
            with tqdm(total=len(results)) as pbar:
                results = await asyncio.gather(
                    *[
                        AsyncLLMAPI._run_task_with_progress(task, pbar)
                        for task in results
                    ]
                )
            return results

        idx = 0
        all_df = []
        while idx < len(data):
            file = f"{idx}_{keyword}.csv"
            file_dir = os.path.join(output_dir, file)

            if os.path.exists(file_dir):
                print(f"{file_dir} already exist! Just skip.")
                tmp_df = pd.read_csv(file_dir)
            else:
                tmp_data = data[idx : idx + chunk_size]

                loop = asyncio.get_event_loop()
                tmp_result = loop.run_until_complete(_func(llms=llms, data=tmp_data))
                tmp_result = [item.generations[0][0].text for item in tmp_result]
                tmp_df = pd.DataFrame({"infer": tmp_result})

                # 如果文件夹不存在，则创建
                if not os.path.exists(tmp_folder := os.path.dirname(file_dir)):
                    os.makedirs(tmp_folder)

                tmp_df.to_csv(file_dir, index=False)

            all_df.append(tmp_df)
            idx += chunk_size

        all_df = pd.concat(all_df)
        all_df.to_csv(os.path.join(output_dir, f"all_{keyword}.csv"), index=False)
        return all_df


if __name__ == "__main__":

    # 生成 数学计算数据集

    texts = []
    labels = []

    for _ in range(1000):
        text, label = generate_arithmetic_expression(2)
        texts.append(text)
        labels.append(label)

    llm = AsyncLLMAPI(
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
        api_key="{}".format(os.environ.get("API_KEY", "0")),
    )

    AsyncLLMAPI.async_run(
        [llm], texts, keyword="数学计算", output_dir="output", chunk_size=500
    )

