


import os
import time
from typing import Optional, List
import openai
from openai import OpenAI
# 加载 .env 到环境变量
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())

# 配置 OpenAI 服务

def q2r(input: str) -> str:
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input,
            }
        ],
        model="gpt-3.5-turbo",
    )

    res = chat_completion.choices[0].message.content

    return res

def q2r_plus(
    input: str,
    max_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop_sequences: Optional[List[str]] = None,
    retries: int = 3,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
) -> str:
    """
    新版 OpenAI 接口实现，包含错误重试和动态调整 max_tokens 的问答函数
    
    参数：
    - input: 用户输入的提问内容
    - max_tokens: 最大 token 数（默认 500）
    - temperature: 温度参数（默认 0.7）
    - retries: 最大重试次数（默认 3）
    - 其他参数保持与 OpenAI API 一致
    
    返回：
    - 模型生成的回答字符串
    """
    
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        organization=organization,
    )
    
    target_length = max_tokens
    retry_cnt = 0
    backoff_time = 30
    response = ""
    
    while retry_cnt <= retries:
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": input}],
                max_tokens=target_length,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                n=1
            )
            response = chat_completion.choices[0].message.content
            break
            
        except openai.APIError as e:
            print(f"API 错误: {str(e)}")
            if "Please reduce your prompt" in str(e):
                target_length = int(target_length * 0.8)
                print(f"自动缩减 token 数至 {target_length}...")
            else:
                print(f"{backoff_time} 秒后重试...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
            retry_cnt += 1
            
        except Exception as e:
            print(f"意外错误: {str(e)}")
            print(f"{backoff_time} 秒后重试...")
            time.sleep(backoff_time)
            backoff_time *= 1.5
            retry_cnt += 1
    
    return response
