#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试视频生成失败检测和处理功能
"""

import os
import sys
import json
import tempfile
import shutil

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase2_video_gen import VideoGenerator

def test_failed_task_detection():
    """
    测试失败任务检测功能
    """
    print("🚀 测试视频生成失败检测功能...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 初始化VideoGenerator
        generator = VideoGenerator(
            api_key="fake_key",
            url="https://api.example.com/v1/chat/completions",
            videos_dir=os.path.join(temp_dir, "videos"),
            tasks_dir=os.path.join(temp_dir, "tasks")
        )
        
        # 创建模拟的失败任务结果
        failed_task = {
            "id": "test_task_1",
            "prompt": "Test prompt",
            "filename_prefix": "Test_Object_Surprise_1",
            "status": "failed",
            "video_url": None,
            "save_path": None,
            "created_at": "2023-01-01 12:00:00"
        }
        
        # 测试monitor_tasks_with_progress方法
        print("\n🔍 测试monitor_tasks_with_progress方法...")
        generator.monitor_tasks_with_progress([failed_task], auto_download=False)
        
        # 测试失败任务处理逻辑
        print("\n🔍 测试失败任务处理逻辑...")
        
        # 创建模拟的physibench_surprise_v79_unique.json文件
        test_entries = [
            {
                "constraints": {
                    "keyword": "Test Object",
                    "type": "Surprise Scenario"
                },
                "final_t2v_prompt": "Test prompt"
            },
            {
                "constraints": {
                    "keyword": "Another Object",
                    "type": "Surprise Scenario"
                },
                "final_t2v_prompt": "Another prompt"
            }
        ]
        
        with open(os.path.join(temp_dir, "physibench_surprise_v79_unique.json"), "w", encoding="utf-8") as f:
            json.dump(test_entries, f, ensure_ascii=False, indent=2)
        
        print("✅ 测试完成！")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_failed_task_detection()