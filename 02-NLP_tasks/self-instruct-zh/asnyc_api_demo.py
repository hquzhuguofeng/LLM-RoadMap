import random
import asyncio
from uuid import uuid4
from tqdm import  tqdm
from dataclasses import dataclass
from aiolimiter import AsyncLimiter

'''
    1-模拟API响应时间：使用 random.randint(0, 2) 随机等待0到2秒，模拟真实的API响应时间。
    2-限速：使用 AsyncLimiter 控制每秒最大请求数量，防止因过多请求而被服务器拒绝服务或达到速率限制。
    3-并发处理：使用 asyncio 和 asyncio.gather 实现并发处理，提高效率。
    4-进度跟踪：使用 tqdm 显示任务进度，提供友好的用户体验。
'''

# 创建限速器，每秒最多发出5个请求
limiter = AsyncLimiter(5, 1)

@dataclass
class Token:
    uid : str
    idx : int
    cnt : int = 0

# 将connect_web 改为异步函数
async def llm_api(data):
    t = random.randint(0, 2)
    # 模拟api调用
    await asyncio.sleep(t)
    return data * 10

# 增加令牌的计数器，使用限速器确保不超过每秒5个请求，添加额外的延迟以模拟实际网络延迟
async def call_api(token, data, rate_limit_seconds=0.5):
    token.cnt += 1
    async with limiter:
        await asyncio.sleep(rate_limit_seconds)
        return await llm_api(data)

# 等待任务完成并更新进度条 
async def _run_task_with_progress(task, par):
    result = await task
    par.update(1)
    return result

async def main():
    workers = 1
    tokens = [Token(uid=str(uuid4()), idx = i) for i in range(workers)]

    nums = 100
    data = [i for i in range(nums)]
    results = [call_api(tokens[int(i % workers)], item) for i, item in enumerate(data)]

    with tqdm(total=len(results)) as pbar:
        results = await asyncio.gather(
            *(_run_task_with_progress(task=task, par=pbar) for task in results)
        )
    return results

result = asyncio.run(main())
print(result)