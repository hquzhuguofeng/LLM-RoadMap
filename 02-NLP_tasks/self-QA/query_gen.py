import os
from prompt import Query_Instruction
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import re
import time
import json
from tqdm import tqdm
from retrival_doc import get_chunk_retrival
from mydocloader import RapidOCRDocLoader
from async_llm_api_batch_data import AsyncLLMAPI
from dataclasses import dataclass, field
import asyncio
import sys
from aiolimiter import AsyncLimiter
from retrival_doc import format_docs,template

os.environ["OPENAI_API_KEY"] = "None"

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def get_all_file(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def get_file_content(file_path):
    loader = RapidOCRDocLoader(file_path=file_path)
    docs = loader.load()
    return docs

def clean_data(input_text):
    """
    清理输入文本中的不合理换行符和多余空白字符。
    
    参数:
        input_text (str): 输入的原始文本数据。
    
    返回:
        str: 清理后的文本数据。
    """
    cleaned_text = re.sub(r'\n+', '\n', input_text)
    cleaned_text = re.sub(r'\s+', '', cleaned_text).strip()
    cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', cleaned_text)
    return cleaned_text

@dataclass
class CustomAsyncLLMAPI(AsyncLLMAPI):
    max_concurrency: int = field(default=70)  # 最大并发数
    max_rate: float = field(default=1.0)      # 每秒最大请求数

    def __post_init__(self):
        super().__post_init__()
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.limiter = AsyncLimiter(max_rate=self.max_rate)

    async def __call__(self, text):
        self.cnt += 1
        async with self.semaphore:
            async with self.limiter:
                response = await self.llm.agenerate([text])
                return response.generations[0][0].text
            
    async def chat_llm_async(self, prompt):
        async with self.semaphore:
            async with self.limiter:
                ai_message = await self.llm.agenerate(
                    [prompt],
                    temperature=0.1,  # 设置温度
                    max_tokens=300,  # 设置最大 token 长度
                )
                return ai_message.generations[0][0].text
            
    async def _get_ans(self, retriver, query_list):
        tasks = []
        for query in query_list:
            task = self.process_query(retriver, query)
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results

    async def process_query(self, retriver, query):
            refs = retriver.invoke(query)
            context = format_docs(refs)
            prompt = template.format(context=context, question=query)
            pred = await self.chat_llm_async(prompt)
            return pred

    async def custom_async_run(
        self, 
        dir_path: str, 
        chun_size: int = 2000, 
        query_instruction: str = Query_Instruction,
        output_name: str = ''
    ) -> None:
        file_list = get_all_file(dir_path)
        for file_path in tqdm(file_list, desc="query_gen", total=len(file_list)):
            print("file_path: ", file_path)
            file_content = get_file_content(file_path)[0].page_content
            _file_content = clean_data(file_content)
            chunk_list = [
                _file_content[i:i+chun_size] for i in range(0, len(_file_content), chun_size)
            ]
            print("chunk_list: ", len(chunk_list))

            tasks = []
            for idx, chunk in enumerate(chunk_list[:12]):
                if len(chunk) < 300:
                    continue
                query = query_instruction.replace("{text}", chunk)
                tasks.append(self.__call__(query))
        
            responses = await asyncio.gather(*tasks)
            for idx, response in tqdm(enumerate(responses), desc="answering", total=len(responses)):
                pattern = r'"query": "(.*?)"'
                matches = re.findall(pattern, response)
                query_list = []
                try:
                    for match in matches:
                        query_list.append(match)
                except Exception as e:
                    print(f"error: llm_output - {e}")

                retriever = get_chunk_retrival(chunk, file_path)
                pred_list = await self._get_ans(retriever, query_list)

                if len(query_list) > 0:
                    with open(f"./output/{output_name}.jsonl", "a", encoding='utf-8') as f:
                        for i in range(len(query_list)):
                            f.write(json.dumps({"query": query_list[i], "answer": pred_list[i]}, ensure_ascii=False) + "\n")
        print("all done!")

if __name__ == '__main__':
    chunk_size = 2000
    llm = CustomAsyncLLMAPI(
        base_url='http://192.168.202.123:8756/v1',
        api_key='',
        model='deepseek32b',
        max_concurrency=100,  # 设置最大并发数
        max_rate=100         # 设置每秒最大请求数
    )
    asyncio.run(
        llm.custom_async_run(
            dir_path='./data', 
            chun_size=chunk_size,
            output_name='法律法规-chunk{}'.format(chunk_size)
            )
        )



