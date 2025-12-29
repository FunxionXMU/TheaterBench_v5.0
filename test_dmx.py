# -*- coding: utf-8 -*-
"""
sora-2-hd-15s-chat 文生视频 / 对话模型非流式调用示例（支持并行任务）

使用前准备
- 安装依赖：pip install requests tqdm
- 配置密钥：建议从环境变量或安全配置读取（如 DMX_API_KEY），避免硬编码

关键参数说明
- model：对话模型名称（如 sora-2-hd-15s-chat）
- messages：OpenAI 兼容消息结构 [{role, content}]
- stream：是否启用 SSE 流式输出（本示例为False）
- headers：Bearer Token 认证
"""
import requests
import json
import concurrent.futures
import logging
import os
import time
from datetime import datetime
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"dmx_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# 端点：DMX API 的 chat completions（OpenAI 兼容）
url = "https://www.dmxapi.cn/v1/chat/completions"

# API密钥配置（优先从环境变量读取）
API_KEY = os.environ.get('DMX_API_KEY', 'sk-hY1PLRISvYRksP0HNJELF2NIv3oqTeW07wAEO0ak432VHHDf')

# 请求头
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 全局任务进度存储
task_progress_store = {}

# 任务状态常量
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

def create_payload(prompt, model="sora-2-hd-15s-chat"):
    """创建请求体"""
    try:
        return {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
        }
    except Exception as e:
        logger.error(f"创建请求体失败: {e}")
        raise

def call_chat_api(prompt, model="sora-2-hd-15s-chat", timeout=30):
    """以非流式方式调用 DMX Chat Completions 并返回完整响应。
    
    Args:
        prompt: 用户提示词
        model: 使用的模型名称
        timeout: 请求超时时间（秒）
        
    Returns:
        包含结果的字典或None
    """
    task_id = f"{model}_{hash(prompt) % 10000}"
    logger.info(f"任务 {task_id} 开始处理（提示词: {prompt[:30]}...）")
    
    # 初始化任务进度
    task_progress_store[task_id] = {
        "task_id": task_id,
        "prompt": prompt,
        "model": model,
        "status": STATUS_RUNNING,
        "progress": 0,
        "total": 100,
        "start_time": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat(),
        "error": None
    }
    
    start_time = time.time()
    payload = create_payload(prompt, model)
    
    try:
        # 更新进度为20%
        update_task_progress(task_id, 20, "正在发送请求...")
        
        response = requests.post(
            url, 
            headers=headers, 
            json=payload, 
            stream=False,
            timeout=timeout
        )
        
        # 更新进度为50%
        update_task_progress(task_id, 50, "正在处理响应...")
        
        response.raise_for_status()  # 抛出HTTP错误
        
        # 更新进度为70%
        update_task_progress(task_id, 70, "正在解析响应...")
        
        # 解析完整的JSON响应
        response_json = response.json()
        choices = response_json.get("choices", [])
        
        if not choices:
            logger.warning(f"任务 {task_id} 响应中没有choices字段")
            update_task_progress(task_id, 100, "处理失败：无有效响应")
            task_progress_store[task_id]["status"] = STATUS_FAILED
            return None
        
        message = choices[0].get("message", {})
        content = message.get("content", "")
        
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        logger.info(f"任务 {task_id} 处理完成，耗时 {elapsed_time}s，内容长度: {len(content)} 字符")
        
        # 更新进度为100%
        update_task_progress(task_id, 100, "处理完成")
        task_progress_store[task_id]["status"] = STATUS_COMPLETED
        task_progress_store[task_id]["end_time"] = datetime.now().isoformat()
        task_progress_store[task_id]["elapsed_time"] = elapsed_time
        
        return {
            "task_id": task_id,
            "prompt": prompt,
            "content": content,
            "full_response": response_json,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except requests.exceptions.Timeout:
        logger.error(f"任务 {task_id} 请求超时（{timeout}秒）")
        logger.debug(f"超时详情 - URL: {url}, 模型: {model}, 提示词长度: {len(prompt)}")
        
        # 更新任务状态为失败
        update_task_progress(task_id, 100, f"请求超时（{timeout}秒）")
        task_progress_store[task_id]["status"] = STATUS_FAILED
        task_progress_store[task_id]["error"] = "request_timeout"
        task_progress_store[task_id]["end_time"] = datetime.now().isoformat()
        
        return {
            "task_id": task_id,
            "prompt": prompt,
            "error": "request_timeout",
            "error_message": f"请求超时（{timeout}秒），API可能响应慢或服务不可用",
            "timestamp": datetime.now().isoformat()
        }
    except requests.exceptions.ConnectionError as e:
        logger.error(f"任务 {task_id} 连接错误: {str(e)}")
        
        # 更新任务状态为失败
        update_task_progress(task_id, 100, f"连接失败: {str(e)}")
        task_progress_store[task_id]["status"] = STATUS_FAILED
        task_progress_store[task_id]["error"] = "connection_error"
        task_progress_store[task_id]["end_time"] = datetime.now().isoformat()
        
        return {
            "task_id": task_id,
            "prompt": prompt,
            "error": "connection_error",
            "error_message": f"连接失败，请检查网络或API地址: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else "未知"
        error_details = ""
        try:
            if e.response:
                # 获取更详细的错误信息
                error_details = e.response.text
                # 尝试解析JSON错误
                try:
                    err_json = e.response.json()
                    error_details = json.dumps(err_json, ensure_ascii=False)
                except json.JSONDecodeError:
                    # 如果不是JSON，使用原始文本
                    pass
            else:
                error_details = str(e)
        except Exception as parse_err:
            error_details = f"无法解析错误响应: {parse_err}"
        
        # 特殊处理403错误
        error_msg = f"HTTP错误 {status_code}: {error_details}"
        if status_code == 403:
            error_msg += " (可能是API密钥无效或权限不足，请检查API密钥配置)"
            logger.error(f"任务 {task_id} {error_msg}")
        else:
            logger.error(f"任务 {task_id} {error_msg}")
        
        # 更新任务状态为失败
        update_task_progress(task_id, 100, error_msg)
        task_progress_store[task_id]["status"] = STATUS_FAILED
        task_progress_store[task_id]["error"] = "http_error"
        task_progress_store[task_id]["end_time"] = datetime.now().isoformat()
        
        return {
            "task_id": task_id,
            "prompt": prompt,
            "error": "http_error",
            "error_message": error_msg,
            "status_code": status_code,
            "response_headers": dict(e.response.headers) if e.response else {},
            "timestamp": datetime.now().isoformat()
        }
    except json.JSONDecodeError as e:
        logger.error(f"任务 {task_id} 解析响应失败: {e}")
        
        # 更新任务状态为失败
        update_task_progress(task_id, 100, f"解析失败: {e}")
        task_progress_store[task_id]["status"] = STATUS_FAILED
        task_progress_store[task_id]["error"] = "json_decode_error"
        task_progress_store[task_id]["end_time"] = datetime.now().isoformat()
        
        return {
            "task_id": task_id,
            "prompt": prompt,
            "error": "json_decode_error",
            "error_message": f"解析响应失败: {e}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"任务 {task_id} 发生未知错误: {e}", exc_info=True)
        
        # 更新任务状态为失败
        update_task_progress(task_id, 100, f"未知错误: {e}")
        task_progress_store[task_id]["status"] = STATUS_FAILED
        task_progress_store[task_id]["error"] = "unknown_error"
        task_progress_store[task_id]["end_time"] = datetime.now().isoformat()
        
        return {
            "task_id": task_id,
            "prompt": prompt,
            "error": "unknown_error",
            "error_message": f"未知错误: {e}",
            "timestamp": datetime.now().isoformat()
        }

def process_single_task(prompt, model="sora-2-hd-15s-chat", timeout=30):
    """处理单个任务"""
    return call_chat_api(prompt, model, timeout)

def process_parallel_tasks(prompts, model="sora-2-hd-15s-chat", max_workers=5, timeout=30, retry_count=1):
    """并行处理多个任务
    
    Args:
        prompts: 提示词列表
        model: 使用的模型名称
        max_workers: 最大并行工作线程数
        timeout: 每个请求的超时时间（秒）
        retry_count: 失败时的重试次数
        
    Returns:
        包含所有任务结果的列表，包括成功和失败的任务
    """
    results = []
    failed_tasks = []
    
    total_tasks = len(prompts)
    logger.info(f"开始并行处理 {total_tasks} 个任务，最大并行数: {max_workers}，超时设置: {timeout}s，重试次数: {retry_count}")
    
    start_time = time.time()
    
    # 第一次处理所有任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_prompt = {
            executor.submit(call_chat_api, prompt, model, timeout): prompt 
            for prompt in prompts
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    # 检查是否有错误
                    if "error" in result:
                        failed_tasks.append(prompt)
                else:
                    failed_tasks.append(prompt)
                    logger.warning(f"任务返回None: {prompt[:30]}...")
            except Exception as e:
                failed_tasks.append(prompt)
                logger.error(f"获取任务结果时出错: {e}", exc_info=True)
    
    # 重试失败的任务
    for retry in range(retry_count):
        if not failed_tasks:
            break
        
        logger.info(f"开始第 {retry + 1} 次重试，剩余失败任务: {len(failed_tasks)}")
        current_failed = failed_tasks.copy()
        failed_tasks.clear()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(current_failed))) as executor:
            future_to_prompt = {
                executor.submit(call_chat_api, prompt, model, timeout): prompt 
                for prompt in current_failed
            }
            
            for future in concurrent.futures.as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    if result:
                        # 更新现有结果或添加新结果
                        existing_index = next(
                            (i for i, r in enumerate(results) if r.get("prompt") == prompt), 
                            None
                        )
                        if existing_index is not None:
                            results[existing_index] = result
                        else:
                            results.append(result)
                        
                        # 检查是否仍有错误
                        if "error" in result:
                            failed_tasks.append(prompt)
                    else:
                        failed_tasks.append(prompt)
                except Exception as e:
                    failed_tasks.append(prompt)
                    logger.error(f"重试任务出错: {e}")
    
    end_time = time.time()
    total_elapsed_time = round(end_time - start_time, 2)
    
    # 统计结果
    success_count = sum(1 for r in results if "error" not in r)
    failed_count = len(failed_tasks)
    
    logger.info(
        f"并行任务处理完成，总耗时: {total_elapsed_time}s，" 
        f"成功: {success_count}，失败: {failed_count}，总任务数: {total_tasks}"
    )
    
    return results

def save_results_to_json(results, filename=None):
    """将结果保存到JSON文件"""
    if not results:
        logger.warning("没有结果可保存")
        return None
    
    if not filename:
        filename = f"dmx_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # 过滤敏感信息
            safe_results = []
            for result in results:
                safe_result = result.copy()
                # 移除可能包含敏感信息的完整响应
                if "full_response" in safe_result:
                    safe_result.pop("full_response")
                safe_results.append(safe_result)
            
            json.dump(safe_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {filename}")
        return filename
    except Exception as e:
        logger.error(f"保存结果失败: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # 示例：并行处理多个提示词
    prompts = [
        "咖啡店内工作场景,生成视频比例16:9",
        "猫咪玩耍场景,生成视频比例1:1"
    ]
    
    # 配置参数
    config = {
        "model": "sora-2-hd-15s-chat",
        "max_workers": 3,
        "timeout": 60,  # 增加超时时间
        "retry_count": 1,
        "run_parallel": True,
        "save_results": True
    }
    
    logger.info(f"配置信息: {json.dumps(config, ensure_ascii=False)}")
    
    if config["run_parallel"]:
        # 并行运行多个任务
        results = process_parallel_tasks(
            prompts, 
            model=config["model"],
            max_workers=config["max_workers"],
            timeout=config["timeout"],
            retry_count=config["retry_count"]
        )
    else:
        # 串行运行示例
        results = []
        for prompt in prompts:
            result = process_single_task(prompt, model=config["model"], timeout=config["timeout"])
            if result:
                results.append(result)
    
    # 保存结果
    if config["save_results"]:
        save_results_to_json(results)
    
    # 打印结果摘要
    print("\n=== 结果摘要 ===")
    for i, result in enumerate(results):
        print(f"\n任务 {i+1}:")
        print(f"提示词: {result['prompt']}")
        if "error" in result:
            print(f"状态: 失败")
            print(f"错误类型: {result['error']}")
            print(f"错误信息: {result['error_message']}")
        else:
            print(f"状态: 成功")
            print(f"耗时: {result['elapsed_time']}s")
            print(f"响应前100字符: {result['content'][:100]}...")