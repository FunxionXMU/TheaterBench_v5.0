#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证phase5_video_understanding_test.py的修改是否正常工作
"""

import json
import os

# 读取测试结果文件
TEST_FILE = 'video_understanding_test_v79.json'

def test_transformation_type_field():
    """测试结果文件中是否包含transformation_type字段"""
    if not os.path.exists(TEST_FILE):
        print(f"❌ 文件 {TEST_FILE} 不存在")
        return False
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_results = json.load(f)
    
    print(f"📊 读取到 {len(test_results)} 条测试结果")
    
    # 检查前10条结果是否包含transformation_type字段
    has_transformation_type = True
    for i, result in enumerate(test_results[:10]):
        if 'transformation_type' not in result:
            print(f"❌ 第 {i+1} 条结果缺少 transformation_type 字段")
            has_transformation_type = False
        else:
            print(f"✅ 第 {i+1} 条结果包含 transformation_type: {result['transformation_type']}")
    
    return has_transformation_type

def main():
    print("🔍 开始测试...")
    has_transformation_type = test_transformation_type_field()
    
    if has_transformation_type:
        print("\n🎉 测试通过：所有结果都包含 transformation_type 字段")
    else:
        print("\n❌ 测试失败：部分结果缺少 transformation_type 字段")

if __name__ == "__main__":
    main()