import requests
import json
import time
import sys
import threading
from typing import Dict, Any, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import datetime
# 导入prompt解析模块
from prompt_parser import get_prompts, jsonl_file

class SoraVideoBatchGenerator:
    def __init__(self, api_key: str, base_url: str = "https://www.dmxapi.cn/v1/chat/completions"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        # 生成带时间戳的文件名，格式：sora_tasks_年月日时分秒.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tasks_file = f"sora_tasks_{timestamp}.json"
        self.videos_dir = "videos"
        
        # 确保videos目录存在
        if not os.path.exists(self.videos_dir):
            os.makedirs(self.videos_dir)
            print(f"已创建视频保存目录: {self.videos_dir}")
    
    def submit_video_task(self, prompt: str, task_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        提交视频生成任务（不等待完成）
        
        Args:
            prompt: 视频描述提示词
            task_id: 自定义任务ID（可选）
            **kwargs: 其他参数
        
        Returns:
            包含任务ID的响应
        """
        # 构建请求数据
        payload = {
            "model": "sora-2-hd-10s-chat",
            "prompt": prompt,
            "aspectRatio": kwargs.get("aspect_ratio", "16:9"),
            "duration": kwargs.get("duration", 10),
            "remixTargetId": kwargs.get("remix_target_id", ""),
            "characters": kwargs.get("characters", []),
            "size": kwargs.get("size", "small"),
            "webHook": "-1",  # 设置为-1，使用轮询方式获取结果
            "shutProgress": kwargs.get("shut_progress", False)
        }
        
        if kwargs.get("url"):
            payload["url"] = kwargs.get("url")
        
        print(f"提交任务请求数据: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/video/sora-video",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            print(f"提交任务响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            
            if response.status_code != 200:
                raise Exception(f"服务器返回错误: {response.status_code}")
            
            # 尝试解析响应
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                print(f"响应不是有效的JSON格式: {e}")
                print(f"原始响应: {response.text}")
                raise Exception(f"响应解析失败: {response.text}")
            
            if result.get("code") != 0:
                raise Exception(f"API返回错误: {result.get('msg', 'Unknown error')}")
            
            # 返回任务ID
            task_id = result["data"]["id"]
            print(f"视频生成任务已提交，任务ID: {task_id}")
            
            return {
                "task_id": task_id,
                "prompt": prompt,
                "status": "submitted",
                "submitted_at": datetime.now().isoformat(),
                "params": kwargs
            }
                
        except Exception as e:
            print(f"提交任务时出错: {str(e)}")
            raise
    
    def batch_submit_tasks(self, prompts: List[str], max_workers: int = 2) -> List[Dict[str, Any]]:
        """
        批量提交视频生成任务
        
        Args:
            prompts: 提示词列表
            max_workers: 最大并发数
        
        Returns:
            任务信息列表
        """
        tasks = []
        
        print(f"开始批量提交 {len(prompts)} 个视频生成任务...")
        print(f"最大并发数: {max_workers}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_prompt = {
                executor.submit(self.submit_video_task, prompt): prompt 
                for prompt in prompts
            }
            
            # 收集结果
            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    task_info = future.result()
                    tasks.append(task_info)
                    print(f"任务提交成功: {task_info['task_id']}")
                except Exception as e:
                    print(f"任务提交失败: {prompt} - 错误: {str(e)}")
        
        return tasks
    
    def save_tasks_to_json(self, tasks: List[Dict[str, Any]], filename: str = None):
        """
        将任务信息保存到JSON文件
        
        Args:
            tasks: 任务信息列表
            filename: 文件名（可选）
        """
        if not filename:
            filename = self.tasks_file
        
        # 准备保存的数据
        data = {
            "saved_at": datetime.now().isoformat(),
            "total_tasks": len(tasks),
            "tasks": tasks
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"任务信息已保存到: {filename}")
        except Exception as e:
            print(f"保存任务信息失败: {str(e)}")
    
    def load_tasks_from_json(self, filename: str = None) -> Dict[str, Any]:
        """
        从JSON文件加载任务信息
        
        Args:
            filename: 文件名（可选）
        
        Returns:
            任务信息
        """
        if not filename:
            # 列出所有sora_tasks_开头的JSON文件
            task_files = []
            try:
                for file in os.listdir('.'):
                    if file.startswith('sora_tasks_') and file.endswith('.json'):
                        task_files.append(file)
                
                # 按修改时间排序，最新的文件排在前面
                task_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                if task_files:
                    print("\n请选择要查看的任务文件:")
                    for i, file in enumerate(task_files, 1):
                        # 获取文件的修改时间
                        mod_time = datetime.fromtimestamp(os.path.getmtime(file))
                        mod_time_str = mod_time.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"{i}. {file}  (修改时间: {mod_time_str})")
                    
                    # 让用户选择
                    try:
                        choice = input("请输入文件序号 (默认选择最新的[1]): ").strip()
                        if not choice or not choice.isdigit():
                            choice = 1  # 默认选择第一个（最新的）
                        else:
                            choice = int(choice)
                        
                        if 1 <= choice <= len(task_files):
                            filename = task_files[choice - 1]
                            print(f"\n已选择文件: {filename}")
                        else:
                            print("无效的选择，使用默认文件")
                            filename = self.tasks_file
                    except Exception:
                        print("输入错误，使用默认文件")
                        filename = self.tasks_file
                else:
                    print("没有找到sora_tasks_开头的JSON文件，使用默认文件")
                    filename = self.tasks_file
            except Exception as e:
                print(f"获取文件列表失败: {str(e)}")
                filename = self.tasks_file
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"已从 {filename} 加载 {len(data.get('tasks', []))} 个任务")
            return data
        except Exception as e:
            print(f"加载任务信息失败: {str(e)}")
            return {}
    
    def check_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        检查单个任务状态
        
        Args:
            task_id: 任务ID
        
        Returns:
            任务状态信息
        """
        payload = {"id": task_id}
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/draw/result",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") != 0:
                return {
                    "task_id": task_id,
                    "status": "error",
                    "error": result.get("msg", "Unknown error")
                }
            
            return result["data"]
            
        except Exception as e:
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e)
            }
    
    def is_video_downloaded(self, task_id: str) -> bool:
        """
        检查指定任务ID的视频是否已下载
        
        Args:
            task_id: 任务ID
        
        Returns:
            视频是否已存在
        """
        filename = f"{task_id}.mp4"
        file_path = os.path.join(self.videos_dir, filename)
        return os.path.exists(file_path)
    
    def monitor_tasks_with_progress(self, task_infos: List[Dict[str, Any]], update_interval: int = 10, auto_download: bool = True):
        """
        监控多个任务的状态，显示每个任务的生成进度条
        
        Args:
            task_infos: 任务信息列表
            update_interval: 更新间隔（秒）
            auto_download: 是否自动下载完成的视频
        """
        print(f"开始监控 {len(task_infos)} 个任务...")
        if auto_download:
            print("视频生成完成后将自动下载到 'videos' 文件夹")
            print("将跳过已存在的视频文件")
        print("按 Ctrl+C 停止监控")
        
        # 为每个任务创建进度条
        progress_bars = {}
        for task_info in task_infos:
            task_id = task_info["task_id"]
            prompt_preview = task_info["prompt"][:30] + "..." if len(task_info["prompt"]) > 30 else task_info["prompt"]
            progress_bars[task_id] = tqdm(
                total=100,
                desc=f"任务 {task_id[:8]}",
                bar_format="{l_bar}{bar}| {n:.0f}%/{total}% [{elapsed}<{remaining}] {desc}",
                position=len(progress_bars)
            )
            progress_bars[task_id].set_description_str(f"任务 {task_id[:8]} - {prompt_preview}")
        
        completed_tasks = set()
        task_results = {}  # 存储任务结果
        
        try:
            while len(completed_tasks) < len(task_infos):
                for task_info in task_infos:
                    task_id = task_info["task_id"]
                    
                    if task_id in completed_tasks:
                        continue
                    
                    status_info = self.check_task_status(task_id)
                    status = status_info.get("status", "unknown")
                    progress = status_info.get("progress", 0)
                    
                    # 更新进度条
                    bar = progress_bars[task_id]
                    
                    # 更新进度
                    if progress > bar.n:
                        bar.update(progress - bar.n)
                    
                    # 更新描述
                    prompt_preview = task_info["prompt"][:30] + "..." if len(task_info["prompt"]) > 30 else task_info["prompt"]
                    bar.set_description_str(f"任务 {task_id[:8]} - {prompt_preview} ({status})")
                    
                    # 检查任务是否完成
                    if status in ["succeeded", "failed", "error"]:
                        completed_tasks.add(task_id)
                        task_results[task_id] = status_info
                        
                        if status == "succeeded":
                            bar.set_description_str(f"任务 {task_id[:8]} - 已完成 ✓")
                            
                            # 自动下载视频
                            if auto_download and status_info.get('results'):
                                for i, result in enumerate(status_info['results']):
                                    video_url = result.get('url')
                                    if video_url:
                                        try:
                                            # 使用任务ID作为文件名
                                            filename = f"{task_id}.mp4"
                                            save_path = os.path.join(self.videos_dir, filename)
                                            
                                            # 检查视频是否已存在
                                            if self.is_video_downloaded(task_id):
                                                print(f"\n视频文件已存在，跳过下载: {filename}")
                                                continue
                                            
                                            print(f"\n开始下载视频: {filename}")
                                            file_path = self.download_video(video_url, save_path)
                                            print(f"视频已保存到: {file_path}")
                                        except Exception as e:
                                            print(f"下载失败: {e}")
                        else:
                            bar.set_description_str(f"任务 {task_id[:8]} - 失败 ✗")
                            failure_reason = status_info.get('failure_reason', '') or status_info.get('error', 'Unknown error')
                            print(f"\n任务 {task_id} 失败原因: {failure_reason}")
                
                # 如果所有任务都已完成，退出循环
                if len(completed_tasks) == len(task_infos):
                    print("\n所有任务已完成!")
                    break
                
                # 等待下一次更新
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n监控已停止")
        finally:
            # 关闭所有进度条
            for bar in progress_bars.values():
                bar.close()
            
            # 显示最终结果
            print("\n任务完成情况:")
            succeeded_count = 0
            downloaded_count = 0
            skipped_count = 0
            
            for task_info in task_infos:
                task_id = task_info["task_id"]
                status_info = task_results.get(task_id, {})
                status = status_info.get("status", "unknown")
                print(f"任务 {task_id}: {status}")
                
                if status == "succeeded":
                    succeeded_count += 1
                    if self.is_video_downloaded(task_id):
                        downloaded_count += 1
                    else:
                        skipped_count += 1
            
            print(f"\n成功生成 {succeeded_count}/{len(task_infos)} 个视频")
            print(f"已下载 {downloaded_count} 个视频")
            if skipped_count > 0:
                print(f"跳过 {skipped_count} 个已存在的视频")
    
    def download_video(self, video_url: str, save_path: str = None) -> str:
        """
        下载生成的视频
        
        Args:
            video_url: 视频URL
            save_path: 保存路径（可选）
        
        Returns:
            视频文件路径
        """
        if not save_path:
            # 生成默认文件名
            filename = f"sora_video.mp4"
            save_path = os.path.join(self.videos_dir, filename)
        
        print(f"开始下载视频到: {save_path}")
        
        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            
            # 下载并显示进度条
            with open(save_path, 'wb') as file, tqdm(
                desc="下载进度",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            
            print(f"视频下载完成: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"下载视频失败: {str(e)}")
            raise


def main():
    # 配置您的API密钥
    API_KEY = "sk-hY1PLRISvYRksP0HNJELF2NIv3oqTeW07wAEO0ak432VHHDf"
    
    # 创建视频生成器实例
    generator = SoraVideoBatchGenerator(API_KEY)
    
    # 询问用户想要生成的视频数量
    default_count = 3
    try:
        count_input = input(f"请输入想要生成的视频数量（默认: {default_count}）: ").strip()
        prompt_count = int(count_input) if count_input else default_count
        if prompt_count <= 0:
            print("数量必须大于0，使用默认值3")
            prompt_count = default_count
    except ValueError:
        print("无效输入，使用默认值3")
        prompt_count = default_count
    
    # 从JSONL文件中获取指定数量的prompt
    prompts = get_prompts(jsonl_file, count=prompt_count)
    
    # 如果获取失败或数量不足，使用默认prompt
    if not prompts:
        prompts = [
            "A penguin walking heavily through a hot, shimmering sandy desert with pyramids in the background. Heat waves are visible.",
            "A serene underwater scene with colorful coral reefs, tropical fish swimming around, and sunbeams filtering through the water."
        ]
    
    # 在所有prompt后面添加". No narrator"
    prompts = [f"{prompt}. No narrator!" for prompt in prompts]
    
    print("批量视频生成工具")
    print("=" * 50)
    print(f"准备生成 {len(prompts)} 个视频")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")
    
    print("\n选项:")
    print("1. 提交新任务")
    print("2. 监控现有任务")
    print("3. 查看任务结果")
    
    choice = input("请选择 (1/2/3): ").strip()
    
    try:
        if choice == "1":
            # 提交新任务
            tasks = generator.batch_submit_tasks(prompts, max_workers=2)
            
            if tasks:
                # 保存任务信息到JSON文件
                generator.save_tasks_to_json(tasks)
                
                # 询问是否开始监控（默认自动下载）
                monitor_input = input("是否开始监控任务状态? (y/n, 默认y): ").strip().lower()
                if monitor_input in ['y', 'yes', '']:  # 空输入视为y
                    generator.monitor_tasks_with_progress(tasks, auto_download=True)
            
        elif choice == "2":
            # 监控现有任务
            data = generator.load_tasks_from_json()
            if data and "tasks" in data:
                # 询问是否自动下载（默认是）
                auto_download_input = input("视频生成完成后是否自动下载? (y/n, 默认y): ").strip().lower()
                auto_download = auto_download_input in ['y', 'yes', '']
                
                generator.monitor_tasks_with_progress(data["tasks"], auto_download=auto_download)
            else:
                print("没有找到任务信息，请先提交任务")
                
        elif choice == "3":
            # 查看任务结果
            data = generator.load_tasks_from_json()
            if data and "tasks" in data:
                for i, task in enumerate(data["tasks"], 1):
                    print(f"\n任务 {i}:")
                    print(f"  任务ID: {task.get('task_id')}")
                    print(f"  提示词: {task.get('prompt')}")
                    print(f"  状态: {task.get('status', 'unknown')}")
                    print(f"  提交时间: {task.get('submitted_at')}")
                    
                    # 检查最新状态
                    status_info = generator.check_task_status(task["task_id"])
                    print(f"  最新状态: {status_info.get('status', 'unknown')}")
                    print(f"  进度: {status_info.get('progress', 0)}%")
                    
                    # 检查视频是否已存在
                    if generator.is_video_downloaded(task["task_id"]):
                        print(f"  视频文件已存在: {task['task_id']}.mp4")
                    
                    # 如果有结果，显示视频信息
                    if status_info.get('status') == 'succeeded' and status_info.get('results'):
                        for j, result in enumerate(status_info['results']):
                            print(f"  视频 {j+1}: {result.get('url')}")
                            
                            # 检查视频是否已存在
                            if generator.is_video_downloaded(task["task_id"]):
                                print(f"  视频文件已存在，跳过下载")
                                continue
                            
                            # 默认下载视频
                            download_input = input(f"  是否下载此视频? (y/n, 默认y): ").strip().lower()
                            if download_input in ['y', 'yes', '']:  # 空输入视为y
                                # 使用任务ID作为文件名
                                filename = f"{task['task_id']}.mp4"
                                save_path = os.path.join(generator.videos_dir, filename)
                                
                                try:
                                    file_path = generator.download_video(result.get('url'), save_path)
                                    print(f"  视频已保存到: {file_path}")
                                except Exception as e:
                                    print(f"  下载失败: {e}")
            else:
                print("没有找到任务信息")
                
        else:
            print("无效选择!")
            
    except Exception as e:
        print(f"操作失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()