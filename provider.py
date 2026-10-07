import os
import random
import time

import openai
from openai import OpenAI

from runtime import success_result, failure_result
from shape import to_dict


# Delay credential validation until a real model request is made.
client = None


def get_client():
    global client
    if client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY_PythonAgentPractice")
        if not api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY before making a real model request")
        client = OpenAI(api_key=api_key, base_url=os.getenv("AGENT_BASE_URL", "https://api.deepseek.com"))
    return client


# 每一轮的重试延迟：指数退避 + 随机抖动，避免多个请求同时重试撞在一起
def calculate_retry_delay(attempt, retry_delay):
    base_delay = retry_delay * (2 ** attempt)
    jitter = random.uniform(0, 0.5)
    return base_delay + jitter


# 重试判断函数
# 当前规则仅为状态码集合判断，逻辑简单且只在一处使用
# 这一层抽象暂时收益有限，若后续规则复杂化，可以继续扩展成为retry policy
def is_retryable_status(status_code):
    return status_code in [500, 502, 503, 504]


def call_model(messages, tools=None, max_retries=2, retry_delay=1):
    for attempt in range(max_retries + 1):
        try:
            request_args = {
                "model": os.getenv("AGENT_MODEL", "deepseek-v4-flash"),
                "messages": messages
            }

            if tools:
                request_args["tools"] = tools

            response = get_client().chat.completions.create(**request_args)
            message = response.choices[0].message

            return success_result(to_dict(message))

        except (openai.APIConnectionError, openai.RateLimitError) as err:
            if attempt == max_retries:
                return failure_result(f"Error: Model provider temporarily unavailable: {err}")

        except openai.APIStatusError as err:
            status_code = err.status_code

            if is_retryable_status(status_code):
                if attempt == max_retries:
                    return failure_result(f"Error: Model provider returned status {status_code}")
            else:
                return failure_result(f"Error: Model provider returned status {status_code}")

        call_delay = calculate_retry_delay(attempt, retry_delay)
        time.sleep(call_delay)
