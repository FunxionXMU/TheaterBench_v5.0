import requests
import json
import time
import sys
import threading
import re
from typing import Dict, Any, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import datetime

class SoraVideoBatchGenerator:
    def __init__(self, api_key: str, base_url: str = "https://www.dmxapi.cn"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        # 生成带时间戳的文件名，格式：sora_tasks_年月日时分秒.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tasks_file = f"sora_tasks_{timestamp}.json"
        self.videos_dir = "t2v_videos"
        
        # 确保videos目录存在
        if not os.path.exists(self.videos_dir):
            os.makedirs(self.videos_dir)
            print(f"已创建视频保存目录: {self.videos_dir}")
    
    def load_tasks_from_json(self, tasks_file: str) -> List[Dict[str, Any]]:
        """
        从JSON文件加载任务信息
        """
        try:
            with open(tasks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("tasks", [])
        except Exception as e:
            print(f"❌ 加载任务文件失败: {e}")
            return []
    
    def list_task_files(self) -> List[str]:
        """
        列出所有已保存的任务文件
        """
        task_files = [f for f in os.listdir('.') if f.startswith('sora_tasks_') and f.endswith('.json')]
        # 按时间戳降序排序
        task_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return task_files
    
    def download_all_completed_videos(self, tasks: List[Dict[str, Any]]):
        """
        下载所有已完成任务的视频（跳过已存在的文件）
        """
        print(f"📥 开始检查 {len(tasks)} 个任务的视频下载状态...")
        
        for i, task in enumerate(tasks):
            task_id = task.get("task_id")
            if not task_id:
                print(f"📝 任务 {i+1} 缺少task_id，跳过")
                continue
            
            filename_prefix = task.get("params", {}).get("filename_prefix", task_id[:8])
            print(f"\n🔍 检查任务 {i+1}/{len(tasks)}: {filename_prefix}")
            
            # 检查任务状态
            status_info = self.check_task_status(task_id)
            status = status_info.get("status", "unknown")
            
            if status == "succeeded" and status_info.get('results'):
                for j, result in enumerate(status_info['results']):
                    video_url = result.get('url')
                    if video_url:
                        suffix = f"_{j+1}" if len(status_info['results']) > 1 else ""
                        filename = f"{filename_prefix}{suffix}.mp4"
                        save_path = os.path.join(self.videos_dir, filename)
                        
                        if os.path.exists(save_path):
                            print(f"✅ 视频已存在，跳过下载: {save_path}")
                            continue
                            
                        print(f"📥 开始下载: {save_path}")
                        if self.download_video(video_url, save_path):
                            print(f"✅ 下载成功: {save_path}")
                        else:
                            print(f"❌ 下载失败: {save_path}")
            elif status == "processing":
                print(f"⏳ 任务正在处理中，当前进度: {status_info.get('progress', 0)}%")
            elif status in ["failed", "error"]:
                print(f"❌ 任务失败: {status_info.get('error', '未知错误')}")
            else:
                print(f"ℹ️  任务状态: {status}")
    
    def submit_video_task(self, prompt: str, task_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        提交视频生成任务（不等待完成）
        """
        # 构建请求数据
        payload = {
            "model": "sora-2-hd-10s-chat", # 或者 sora-1.0-turbo, 根据实际可用模型调整
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
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"服务器返回错误: {response.status_code} - {response.text}")
            
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                raise Exception(f"响应解析失败: {response.text}")
            
            if result.get("code") != 0:
                raise Exception(f"API返回错误: {result.get('msg', 'Unknown error')}")
            
            # 返回任务ID
            task_id = result["data"]["id"]
            
            return {
                "task_id": task_id,
                "prompt": prompt,
                "status": "submitted",
                "submitted_at": datetime.now().isoformat(),
                "params": kwargs # 关键：保存包括 filename_prefix 在内的参数
            }
                
        except Exception as e:
            raise
    
    def batch_submit_tasks(self, task_requests: List[Dict[str, Any]], max_workers: int = 2) -> List[Dict[str, Any]]:
        """
        批量提交视频生成任务
        """
        tasks = []
        
        print(f"开始批量提交 {len(task_requests)} 个视频生成任务...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_req = {
                executor.submit(self.submit_video_task, **req): req 
                for req in task_requests
            }
            
            # 收集结果
            for future in as_completed(future_to_req):
                req = future_to_req[future]
                try:
                    task_info = future.result()
                    tasks.append(task_info)
                    prefix = req.get('filename_prefix', 'unknown')
                    print(f"✅ 任务提交成功: {prefix} (ID: {task_info['task_id']})")
                except Exception as e:
                    print(f"❌ 任务提交失败: {req.get('filename_prefix')} - 错误: {str(e)}")
        
        return tasks
    
    def save_tasks_to_json(self, tasks: List[Dict[str, Any]], filename: str = None):
        if not filename:
            filename = self.tasks_file
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
    
    def monitor_tasks_with_progress(self, task_infos: List[Dict[str, Any]], update_interval: int = 10, auto_download: bool = True):
        print(f"开始监控 {len(task_infos)} 个任务...")
        print("按 Ctrl+C 停止监控")
        
        progress_bars = {}
        for task_info in task_infos:
            task_id = task_info["task_id"]
            filename_prefix = task_info.get("params", {}).get("filename_prefix", task_id[:8])
            
            progress_bars[task_id] = tqdm(
                total=100,
                desc=f"{filename_prefix}",
                bar_format="{l_bar}{bar}| {n:.0f}% {desc}",
                position=len(progress_bars)
            )
        
        completed_tasks = set()
        
        try:
            while len(completed_tasks) < len(task_infos):
                for task_info in task_infos:
                    task_id = task_info["task_id"]
                    if task_id in completed_tasks: continue
                    
                    status_info = self.check_task_status(task_id)
                    status = status_info.get("status", "unknown")
                    progress = status_info.get("progress", 0)
                    
                    bar = progress_bars[task_id]
                    if progress > bar.n: bar.update(progress - bar.n)
                    
                    filename_prefix = task_info.get("params", {}).get("filename_prefix", task_id)
                    bar.set_description_str(f"{filename_prefix} ({status})")
                    
                    if status in ["succeeded", "failed", "error"]:
                        completed_tasks.add(task_id)
                        
                        if status == "succeeded":
                            bar.set_description_str(f"{filename_prefix} - 完成 ✓")
                            if auto_download and status_info.get('results'):
                                for j, result in enumerate(status_info['results']):
                                    video_url = result.get('url')
                                    if video_url:
                                        suffix = f"_{j+1}" if len(status_info['results']) > 1 else ""
                                        filename = f"{filename_prefix}{suffix}.mp4"
                                        save_path = os.path.join(self.videos_dir, filename)
                                        
                                        if os.path.exists(save_path):
                                            continue 
                                            
                                        self.download_video(video_url, save_path)
                        else:
                            bar.set_description_str(f"{filename_prefix} - 失败 ✗")
                
                if len(completed_tasks) == len(task_infos): break
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n监控已停止")
        finally:
            for bar in progress_bars.values(): bar.close()
    
    def check_task_status(self, task_id: str) -> Dict[str, Any]:
        payload = {"id": task_id}
        try:
            response = requests.post(
                f"{self.base_url}/v1/draw/result",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            if response.status_code != 200: return {"status": "error", "error": f"HTTP {response.status_code}"}
            result = response.json()
            if result.get("code") != 0: return {"status": "error", "error": result.get("msg")}
            return result["data"]
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def download_video(self, video_url: str, save_path: str) -> str:
        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            name = os.path.basename(save_path)
            with open(save_path, 'wb') as file, tqdm(
                desc=f"下载 {name}", total=total_size, unit='iB', unit_scale=True, leave=False
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            return save_path
        except Exception as e:
            print(f"下载失败 {save_path}: {e}")
            return None

def sanitize_filename(name):
    name = re.sub(r'[\/*?:"<>|]', "", name)
    return name.replace(" ", "_")

def load_evaluation_results():
    """
    加载最新的评估结果文件
    """
    # 查找最新的评估结果文件
    eval_files = [f for f in os.listdir('.') if f.startswith('physibench_evaluated_') and f.endswith('.json')]
    if not eval_files:
        return None
    
    # 按时间戳降序排序，获取最新的文件
    eval_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_eval_file = eval_files[0]
    
    print(f"📊 读取最新评估结果文件: {latest_eval_file}")
    
    try:
        with open(latest_eval_file, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)
        return eval_results
    except Exception as e:
        print(f"❌ 读取评估结果文件失败: {e}")
        return None

def get_prompt_priority(entry, existing_count, eval_results_dict):
    """
    获取prompt的优先级
    优先级：1、已有视频但评估不合格；2、已有视频但分数<9.0；3、还未有视频
    """
    constraints = entry.get('constraints', {})
    keyword = constraints.get('keyword', '')
    s_type = constraints.get('type', '')
    
    # 生成唯一标识符，与评估结果中的键一致
    key = (keyword, s_type)
    
    # 检查是否有评估结果
    if key in eval_results_dict:
        eval_entry = eval_results_dict[key]
        
        # 情况1：已有视频但评估不合格
        if existing_count > 0 and not eval_entry.get('has_valid_video', False):
            return 1
        
        # 情况2：已有视频但分数<9.0
        if existing_count > 0 and eval_entry.get('best_score', 0) < 9.0:
            return 2
    
    # 情况3：还未有视频或没有评估结果
    return 3

def main():
    API_KEY = "sk-hY1PLRISvYRksP0HNJELF2NIv3oqTeW07wAEO0ak432VHHDf"
    generator = SoraVideoBatchGenerator(API_KEY)
    
    print("=" * 50)
    print("🎬 Sora 批量生成器 - surprise 版")
    print("=" * 50)
    
    # 功能选择：创建新任务或下载已有任务的视频
    print("🔧 功能选项:")
    print("1. 创建新的视频生成任务")
    print("2. 从已创建的任务中下载视频")
    choice = input("请选择功能 (1/2, 默认1): ").strip() or "1"
    print()

    if choice == "2":
        # 从已创建的任务中下载视频
        print("� 列出所有已保存的任务文件:")
        task_files = generator.list_task_files()
        
        if not task_files:
            print("❌ 未找到任何任务文件！")
            return
            
        # 显示任务文件列表
        for i, file in enumerate(task_files):
            file_time = datetime.fromtimestamp(os.path.getmtime(file)).strftime("%Y-%m-%d %H:%M:%S")
            file_size = os.path.getsize(file) / 1024  # KB
            print(f"{i+1}. {file} (创建时间: {file_time}, 大小: {file_size:.2f} KB)")
        
        # 让用户选择一个任务文件
        try:
            file_choice = input(f"请选择任务文件 (1-{len(task_files)}, 默认1): ").strip() or "1"
            file_index = int(file_choice) - 1
            if file_index < 0 or file_index >= len(task_files):
                print("❌ 无效的选择！")
                return
            selected_file = task_files[file_index]
        except ValueError:
            print("❌ 无效的输入！")
            return
            
        print(f"\n📂 加载任务文件: {selected_file}")
        tasks = generator.load_tasks_from_json(selected_file)
        
        if not tasks:
            print("❌ 任务文件中没有有效的任务信息！")
            return
            
        # 下载所有已完成的视频
        generator.download_all_completed_videos(tasks)
        print("\n🎉 视频下载完成！")
        return
    
    # 否则，创建新的视频生成任务
    # �🟢 1. 自动查找最新的 surprise JSON 文件
    # 修改：查找 'physibench_surprise_' 开头的文件
    json_files = [f for f in os.listdir('.') if f.startswith('physibench_surprise_') and f.endswith('.json')]
    json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if not json_files:
        print("❌ 未找到 physibench_surprise_*.json 文件！请先运行 phase1_prompt_gen.py 生成。")
        return
        
    target_file = json_files[0]
    print(f"📂 读取数据文件: {target_file}")
    
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        print(f"📊 原始文件包含 {len(full_data)} 个场景")
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 生成模式选择
    print("\n🎨 生成模式:")
    print("1. 多样模式 - 优先为不同prompt生成视频")
    print("2. 补全模式 - 优先补全同个prompt的2个视频")
    print("3. 默认模式 - 按评估结果优先级生成，同个prompt上限6个视频")  # 新增选项
    mode_choice = input("请选择生成模式 (1/2/3, 默认3): ").strip() or "3"
    mode = 'diverse' if mode_choice == "1" else 'complete' if mode_choice == "2" else 'default'
    
    # 2. 设置生成目标
    TARGET_PER_PROMPT = 6 if mode_choice == "3" else 2
    
    # 询问用户要提交的视频数量，默认4个
    max_tasks_input = input("请输入要提交的视频数量 (默认4个): ").strip()
    MAX_TEST_TASKS = int(max_tasks_input) if max_tasks_input.isdigit() else 4
    
    print(f"🎯 目标：确保每个 Prompt 拥有 {TARGET_PER_PROMPT} 个视频")
    print(f"🧪 测试限制：本次运行最多提交 {MAX_TEST_TASKS} 个新任务")
    print()

    # 3. 遍历并检查缺失文件
    task_requests = []
    
    # 统计每个prompt当前已有的视频数量
    prompt_video_counts = []
    for entry in full_data:
        base_prompt = entry.get('final_t2v_prompt', '')
        if not base_prompt: continue
        
        constraints = entry.get('constraints', {})
        keyword = constraints.get('keyword', 'Object')
        s_type = constraints.get('type', 'Scenario')
        
        safe_keyword = sanitize_filename(keyword)
        safe_type = sanitize_filename(s_type.replace(" Scenario", ""))
        
        # 计算当前已有视频数量
        existing_count = 0
        for i in range(TARGET_PER_PROMPT):
            idx = i + 1
            filename_prefix = f"{safe_keyword}_{safe_type}_{idx}"
            expected_filename = f"{filename_prefix}.mp4"
            save_path = os.path.join(generator.videos_dir, expected_filename)
            if os.path.exists(save_path):
                existing_count += 1
        
        prompt_video_counts.append((entry, existing_count, safe_keyword, safe_type))
    
    # 根据模式选择不同的任务生成策略
    if mode == 'diverse':
        # 多样模式：优先为每个prompt生成1个视频，然后再生成更多
        print("🔄 多样模式：优先为不同prompt生成视频")
        
        # 按已有的视频数量排序
        prompt_video_counts.sort(key=lambda x: x[1])
        
        for entry, existing_count, safe_keyword, safe_type in prompt_video_counts:
            if len(task_requests) >= MAX_TEST_TASKS:
                break
                
            base_prompt = entry.get('final_t2v_prompt', '')
            if not base_prompt: continue
            
            # 为当前prompt生成下一个视频
            if existing_count < TARGET_PER_PROMPT:
                idx = existing_count + 1
                filename_prefix = f"{safe_keyword}_{safe_type}_{idx}"
                
                print(f"➕ 添加任务: {filename_prefix}")
                task_requests.append({
                    "prompt": base_prompt,
                    "filename_prefix": filename_prefix,
                    "duration": 5, 
                    "aspect_ratio": "16:9"
                })
    elif mode == 'complete':
        # 补全模式：为每个prompt补全到TARGET_PER_PROMPT个视频
        print("🔄 补全模式：优先补全同个prompt的视频")
        
        for entry in full_data:
            if len(task_requests) >= MAX_TEST_TASKS:
                break

            base_prompt = entry.get('final_t2v_prompt', '')
            if not base_prompt: continue
            
            constraints = entry.get('constraints', {})
            keyword = constraints.get('keyword', 'Object')
            s_type = constraints.get('type', 'Scenario')
            
            safe_keyword = sanitize_filename(keyword)
            safe_type = sanitize_filename(s_type.replace(" Scenario", ""))
            
            final_prompt = base_prompt 
            
            for i in range(TARGET_PER_PROMPT):
                idx = i + 1
                filename_prefix = f"{safe_keyword}_{safe_type}_{idx}"
                expected_filename = f"{filename_prefix}.mp4"
                save_path = os.path.join(generator.videos_dir, expected_filename)
                
                if os.path.exists(save_path):
                    pass
                else:
                    print(f"➕ 添加任务: {filename_prefix}")
                    task_requests.append({
                        "prompt": final_prompt,
                        "filename_prefix": filename_prefix,
                        "duration": 5, 
                        "aspect_ratio": "16:9"
                    })
                    
                    if len(task_requests) >= MAX_TEST_TASKS:
                        print(f"⚠️ 达到测试上限 ({MAX_TEST_TASKS} 个任务)，停止添加。")
                        break
    else:  # 默认模式
        print("🔄 默认模式：按评估结果优先级生成视频")
        
        # 加载评估结果
        eval_results = load_evaluation_results()
        eval_results_dict = {}
        
        # 将评估结果转换为字典以便快速查找
        if eval_results:
            for eval_entry in eval_results:
                keyword = eval_entry.get('object_name', '')
                s_type = eval_entry.get('scenario_type', '')
                eval_results_dict[(keyword, s_type)] = eval_entry
        
        # 计算每个prompt的优先级
        prompt_with_priority = []
        for entry, existing_count, safe_keyword, safe_type in prompt_video_counts:
            priority = get_prompt_priority(entry, existing_count, eval_results_dict)
            prompt_with_priority.append((entry, existing_count, safe_keyword, safe_type, priority))
        
        # 按优先级排序（优先级数值越小越优先），然后按已有视频数量排序
        prompt_with_priority.sort(key=lambda x: (x[4], x[1]))
        
        for entry, existing_count, safe_keyword, safe_type, priority in prompt_with_priority:
            if len(task_requests) >= MAX_TEST_TASKS:
                break
                
            base_prompt = entry.get('final_t2v_prompt', '')
            if not base_prompt: continue
            
            # 为当前prompt生成下一个视频
            if existing_count < TARGET_PER_PROMPT:
                idx = existing_count + 1
                filename_prefix = f"{safe_keyword}_{safe_type}_{idx}"
                
                # 打印优先级信息
                priority_info = "（评估不合格）" if priority == 1 else "（分数<9.0）" if priority == 2 else "（新视频）"
                print(f"➕ 添加任务: {filename_prefix} {priority_info}")
                
                task_requests.append({
                    "prompt": base_prompt,
                    "filename_prefix": filename_prefix,
                    "duration": 5, 
                    "aspect_ratio": "16:9"
                })
        
    if not task_requests:
        print("\n✅ 所有场景都已满足视频数量要求，无需提交新任务！")
        return

    print(f"\n📋 准备提交 {len(task_requests)} 个补全任务")

    confirm = input("🚀 确认开始提交任务? (y/n, 默认y): ").strip().lower()
    if confirm not in ['y', 'yes', '']:
        print("已取消")
        return

    tasks = generator.batch_submit_tasks(task_requests, max_workers=3)
    
    if tasks:
        generator.save_tasks_to_json(tasks)
        print("\n👀 开始监控任务并自动下载...")
        generator.monitor_tasks_with_progress(tasks, auto_download=True)

if __name__ == "__main__":
    main()