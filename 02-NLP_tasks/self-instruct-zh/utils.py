import re
import json
import random
import time
from typing import Union, Dict


'''
这个代码的主要功能包括：

1-生成算术表达式：通过随机选择操作符和数字生成复杂的算术表达式，并计算其结果。
2-解析JSON字符串：从文本中提取并解析第一个有效的JSON对象。
3-计算时间差：计算两个时间点之间的时间差，并格式化输出。
4-记录函数执行时间：通过装饰器记录函数的执行时间，方便性能分析。
'''


def generate_arithmetic_expression(num: int):
    """
    num: 几个操作符
    """
    # 定义操作符和数字范围，除法
    operators = ['+', '-', '*']
    expression = f"{random.randint(1, 100)} {random.choice(operators)} {random.randint(1, 100)}"
    num -= 1
    for _ in range(num):
        expression = f"{expression} {random.choice(operators)} {random.randint(1, 100)}"
    result = eval(expression)
    expression = expression.replace('*', 'x')
    return expression, result

def re_parse_json(text) -> Union[Dict, None]:
    # 提取 JSON 内容
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        json_data = json_match.group(0)
        response_data = json.loads(json_data)
        return response_data
    print(f"异常:\n{text}")
    return None


def calculate_time_difference(start_time, end_time):
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    milliseconds = (elapsed_time - int(elapsed_time)) * 1000

    print(
        f"executed in {int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03} (h:m:s.ms)"
    )


def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行目标函数
        end_time = time.time()  # 记录结束时间

        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        milliseconds = (elapsed_time - int(elapsed_time)) * 1000

        print(
            f"Function '{func.__name__}' executed in {int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03} (h:m:s.ms)")
        return result

    return wrapper


# 测试生成
if __name__ == "__main__":
    expr, res = generate_arithmetic_expression(4)
    print(f"生成的运算表达式: {expr}")
    print(f"计算结果: {res}")

