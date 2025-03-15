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

import asyncio
import os
from dataclasses import dataclass, field
from typing import List, Tuple
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from asyncio import Semaphore

@dataclass
class AsyncLLMBatcher:
    """
    异步高并发LLM调用器，合并优化版本
    
    功能特性：
    - 异步I/O高效并发
    - 自动请求限速控制
    - 错误重试机制
    - 结果顺序保障
    - 分块保存与断点续传
    - 进度条显示
    """
    
    api_key: str
    model: str = "gpt-3.5-turbo"
    system_prompt: str = ""
    temperature: float = 1.0
    max_workers: int = 64          # 最大并发数
    req_per_second: int = 10       # 每秒请求上限
    retry_attempts: int = 3        # 错误重试次数
    chunk_size: int = 500          # 分块处理大小
    timeout: int = 30              # 单请求超时(秒)
    output_dir: str = "output"     # 输出目录
    
    _client: AsyncOpenAI = field(init=False)
    _semaphore: Semaphore = field(init=False)
    _rate_limiter: Semaphore = field(init=False)

    def __post_init__(self):
        self._client = AsyncOpenAI(api_key=self.api_key)
        self._semaphore = Semaphore(self.max_workers)
        self._rate_limiter = Semaphore(self.req_per_second)

    async def _call_api(self, data: Tuple[int, str]) -> Tuple[int, str]:
        """带重试机制的异步请求"""
        idx, text = data
        for _ in range(self.retry_attempts):
            try:
                async with self._rate_limiter:
                    response = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": text}
                            ],
                            temperature=self.temperature
                        ),
                        timeout=self.timeout
                    )
                    return (idx, response.choices[0].message.content)
            except Exception as e:
                print(f"Attempt {_+1} failed: {str(e)}")
        return (idx, None)

    async def _process_chunk(self, chunk: List[Tuple[int, str]]) -> pd.DataFrame:
        """处理单个数据块"""
        tasks = []
        async with self._semaphore:
            for data in chunk:
                task = self._call_api(data)
                tasks.append(task)
            
            results = []
            for future in tqdm_asyncio.as_completed(tasks, desc="Processing"):
                results.append(await future)
            
            # 按原始顺序排序
            results.sort(key=lambda x: x[0])
            return pd.DataFrame(
                data=[res[1] for res in results],
                columns=["response"]
            )

    async def run(self, data: List[str], batch_name: str = "result"):
        """主运行方法"""
        # 准备带索引的数据
        indexed_data = list(enumerate(data))
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 分块处理
        for chunk_idx in range(0, len(data), self.chunk_size):
            chunk = indexed_data[chunk_idx:chunk_idx+self.chunk_size]
            output_path = os.path.join(
                self.output_dir,
                f"{batch_name}_chunk_{chunk_idx//self.chunk_size}.csv"
            )
            
            # 跳过已处理块
            if os.path.exists(output_path):
                print(f"Skipping existing chunk: {output_path}")
                continue
                
            # 处理并保存
            df = await self._process_chunk(chunk)
            df.to_csv(output_path, index=False)
        
        # 合并结果
        return self._merge_results(batch_name)

    def _merge_results(self, batch_name: str) -> pd.DataFrame:
        """合并所有分块结果"""
        all_files = [
            os.path.join(self.output_dir, f)
            for f in os.listdir(self.output_dir)
            if f.startswith(batch_name)
        ]
        
        dfs = []
        for f in sorted(all_files, key=lambda x: int(x.split("_")[-1].split(".")[0])):
            dfs.append(pd.read_csv(f))
        
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(
            os.path.join(self.output_dir, f"FINAL_{batch_name}.csv"),
            index=False
        )
        return merged_df


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

