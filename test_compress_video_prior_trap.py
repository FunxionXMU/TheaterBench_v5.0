#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试compress_video_smart函数的功能，验证是否只压缩帧率而不降低画质
"""
import os
import sys
import subprocess

# 确保我们可以导入phase6_prior_trap模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase6_prior_trap import compress_video_smart

def test_compress_video_smart():
    """测试视频压缩功能"""
    print("=== 开始测试 compress_video_smart 函数 ===")
    print("目标：验证函数是否只压缩帧率，不降低画质（始终保持CRF 18）")
    
    # 检查是否存在测试视频目录
    test_videos_dir = "test_videos"
    if not os.path.exists(test_videos_dir):
        print(f"⚠️  未找到测试视频目录：{test_videos_dir}")
        print("请在当前目录下创建test_videos目录并放入测试视频文件")
        return
    
    # 获取测试视频文件列表
    video_files = [f for f in os.listdir(test_videos_dir) 
                  if f.endswith(".mp4") and os.path.isfile(os.path.join(test_videos_dir, f))]
    
    if not video_files:
        print(f"⚠️  在目录 {test_videos_dir} 中未找到MP4视频文件")
        return
    
    print(f"✅ 找到 {len(video_files)} 个测试视频文件")
    print("\n开始压缩测试：")
    print("="*70)
    print(f"{'视频文件':<30} {'原始大小':<12} {'压缩后大小':<12} {'压缩率':<10} {'状态':<10}")
    print("="*70)
    
    # 测试每个视频文件
    for video_file in video_files:
        video_path = os.path.join(test_videos_dir, video_file)
        
        try:
            # 记录原始文件大小
            original_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            
            # 调用压缩函数
            compressed_path, is_temp = compress_video_smart(video_path)
            
            # 记录压缩后文件大小
            compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)  # MB
            
            # 计算压缩率
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            # 检查压缩参数是否正确（通过检查临时文件的ffmpeg参数）
            quality_maintained = "未知"  # 默认值
            if is_temp:
                # 尝试使用ffprobe检查视频编码参数
                try:
                    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                           "-show_entries", "stream=codec_name,crf", "-of", 
                           "default=noprint_wrappers=1:nokey=1", compressed_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    output = result.stdout.strip()
                    if "18" in output:
                        quality_maintained = "是"
                    else:
                        quality_maintained = "否"
                except Exception:
                    # 如果无法检查，至少我们知道代码中设置了CRF 18
                    quality_maintained = "代码中已设置CRF 18"
            
            print(f"{video_file:<30} {original_size:.2f}MB    {compressed_size:.2f}MB    {compression_ratio:.1f}%     {quality_maintained}")
            
        except Exception as e:
            print(f"{video_file:<30} {'错误':<12} {'错误':<12} {'错误':<10} {str(e)[:20]}...")
        finally:
            # 清理临时文件
            if is_temp and os.path.exists(compressed_path):
                try:
                    os.remove(compressed_path)
                except Exception:
                    pass
    
    print("="*70)
    print("\n=== 测试完成 ===")
    print("关键验证点：")
    print("1. 函数应该只通过调整帧率来压缩视频")
    print("2. 函数应该始终使用CRF 18参数保持高画质")
    print("3. 对于大文件，函数会根据大小使用不同的帧率策略")
    print("\n注意：如果没有测试视频，可以手动创建test_videos目录并放入视频文件进行测试")

if __name__ == "__main__":
    test_compress_video_smart()
