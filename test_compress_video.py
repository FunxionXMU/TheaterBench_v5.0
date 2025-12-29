#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试compress_video_smart函数的功能
"""

import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase5_video_understanding_test import compress_video_smart

def test_compress_video_smart():
    """
    测试视频压缩功能
    """
    print("开始测试compress_video_smart函数...")
    
    # 检查是否存在测试视频文件
    test_videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_videos")
    
    if not os.path.exists(test_videos_dir):
        os.makedirs(test_videos_dir)
        print(f"请将测试视频文件放入目录: {test_videos_dir}")
        print("测试结束：需要测试视频文件")
        return
    
    # 获取目录中的视频文件
    video_files = [f for f in os.listdir(test_videos_dir) 
                  if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    
    if not video_files:
        print(f"在 {test_videos_dir} 目录中未找到视频文件")
        print("测试结束：需要测试视频文件")
        return
    
    # 测试每个视频文件
    for video_file in video_files:
        video_path = os.path.join(test_videos_dir, video_file)
        print(f"\n测试视频: {video_file}")
        
        try:
            # 获取原始文件大小
            original_size = os.path.getsize(video_path) / (1024 * 1024)  # 转换为MB
            print(f"原始文件大小: {original_size:.2f} MB")
            
            # 调用压缩函数
            print("正在压缩视频...")
            compressed_path, is_temp = compress_video_smart(video_path)
            
            # 获取压缩后文件大小
            compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)  # 转换为MB
            print(f"压缩后文件大小: {compressed_size:.2f} MB")
            print(f"压缩率: {(1 - compressed_size/original_size) * 100:.2f}%")
            print(f"压缩后的文件路径: {compressed_path}")
            print(f"是否为临时文件: {is_temp}")
            print("压缩成功！")
            
        except Exception as e:
            print(f"压缩失败: {str(e)}")
    
    print("\n所有测试完成")

if __name__ == "__main__":
    test_compress_video_smart()
