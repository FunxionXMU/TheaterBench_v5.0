import os
import json
import time
from tqdm import tqdm
from phase0_config_data import client_deepseek, HELPER_MODEL
from phase1_prompt_gen import analyzer_agent  # 从phase1_prompt_gen.py导入analyzer_agent

# 复制必要的辅助函数
def clean_and_parse_json(content):
    """
    JSON cleaning function to handle extra data and Markdown issues
    """
    if not content: return None
    content = content.strip()
    
    try:
        if "```" in content:
            first_marker = content.find("```")
            last_marker = content.rfind("```")
            if first_marker != -1 and last_marker != -1 and first_marker != last_marker:
                content = content[first_marker + 3 : last_marker].strip()
                content = content.replace("json", "", 1).strip()
        
        brace_count = 0
        start_idx = -1
        end_idx = -1
        
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx : end_idx + 1]
        
        import re
        content = re.sub(r',\s*([}\]])', r'\1', content)
        
        return json.loads(content)
    except Exception as e:
        print(f"DEBUG: JSON Parse Error - {e}")
        return None

def analyze_missing_entries(data, analyzer_func, file_path=None):
    """
    为缺失transformation_analysis字段的条目调用分析器
    
    Args:
        data: 要处理的数据列表
        analyzer_func: 分析器函数
        file_path: 文件路径，如果提供则实时保存
    
    Returns:
        更新后的数据
    """
    # 统计缺失条目
    missing_entries = [entry for entry in data if 'transformation_analysis' not in entry]
    total = len(missing_entries)
    
    if total == 0:
        print("✅ No entries missing transformation_analysis field")
        return data
    
    # 计数器
    success_count = 0
    error_count = 0
    
    # 使用tqdm显示进度
    for i, entry in enumerate(tqdm(missing_entries, desc="Analyzing entries")):
        try:
            # 提取normal_middle和surreal_middle
            if 'normal_timeline' in entry and entry['normal_timeline'] and 'middle_caption' in entry['normal_timeline']:
                normal_middle = entry['normal_timeline']['middle_caption']
            else:
                normal_middle = None
            
            if 'timeline' in entry and entry['timeline'] and 'middle_caption' in entry['timeline']:
                surreal_middle = entry['timeline']['middle_caption']
            else:
                surreal_middle = None
            
            # 检查是否有足够信息进行分析
            if not normal_middle or not surreal_middle:
                print(f"❌ Skipping entry {i+1}/{total}: Missing normal_middle or surreal_middle")
                error_count += 1
                continue
            
            # 调用分析器函数，最多重试3次
            for retry in range(3):
                try:
                    # 调用分析器
                    result = analyzer_func(normal_middle, surreal_middle)
                    
                    # 验证结果并保存到条目
                    if result and isinstance(result, dict):
                        # 找到对应的原始条目并更新
                        for original_entry in data:
                            if original_entry is entry:
                                original_entry['transformation_analysis'] = result
                                break
                        
                        success_count += 1
                        print(f"✅ Successfully analyzed entry {i+1}/{total} (Retry: {retry})")
                        
                        # 如果提供了文件路径，则每成功分析一个条目就立即保存
                        if file_path:
                            print(f"💾 Saving after successful analysis of entry {i+1}/{total}...")
                            save_json_file(data, file_path)  # 默认backup=False
                        
                        # 添加延迟，避免API调用过于频繁
                        time.sleep(2)
                        break
                    else:
                        print(f"⚠️ Invalid analysis result for entry {i+1}/{total}")
                        if retry < 2:
                            print(f"🔄 Retrying ({retry + 1}/3)...")
                            time.sleep(2)
                except Exception as e:
                    print(f"⚠️ Analysis failed for entry {i+1}/{total}: {str(e)}")
                    if retry < 2:
                        print(f"🔄 Retrying ({retry + 1}/3)...")
                        time.sleep(2)
            else:
                print(f"❌ Failed to analyze entry {i+1}/{total} after 3 retries")
                error_count += 1
        except Exception as e:
            print(f"❌ Unexpected error processing entry {i+1}/{total}: {str(e)}")
            error_count += 1
    
    # 打印总结
    print(f"\n📊 Analysis Summary:")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {error_count}")
    print(f"📈 Success rate: {(success_count / total * 100):.1f}%\n")
    
    return data

def save_json_file(data, file_path, backup=False):
    """
    保存数据到JSON文件，并可选地创建备份
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        backup: 是否创建备份，默认为False
    """
    try:
        # 如果需要备份且文件存在
        if backup and os.path.exists(file_path):
            # 创建带时间戳的备份文件名
            timestamp = int(time.time())
            backup_path = f"{file_path}.backup_{timestamp}"
            
            # 复制文件内容作为备份
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Created backup at {backup_path}")
        
        # 保存数据到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Successfully saved data to {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving file: {str(e)}")
        return False

def read_json_file(file_path):
    """
    读取JSON文件
    
    Args:
        file_path: JSON文件路径
    
    Returns:
        解析后的数据，如果失败则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} entries from {file_path}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in file {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading JSON file: {str(e)}")
        return None

def check_transformation_analysis(data):
    """
    检查数据中缺失transformation_analysis字段的条目数量
    
    Args:
        data: JSON数据列表
    
    Returns:
        缺失字段的条目数量
    """
    return sum(1 for entry in data if 'transformation_analysis' not in entry)

def main():
    """
    主函数，协调整个处理流程
    """
    # 文件路径
    json_file_path = '/Users/tongyao/Documents/VSCode code/v5.2/physibench_surprise_v79_unique.json'
    
    # 首先读取JSON文件
    print("📂 Reading JSON file...")
    data = read_json_file(json_file_path)
    
    if not data:
        print("❌ Failed to read JSON file. Exiting.")
        return
    
    # 在程序开始时创建一次备份
    print("📋 Creating initial backup...")
    save_json_file(data, json_file_path, backup=True)
    
    # 检查缺失transformation_analysis字段的条目
    print("🔍 Checking for entries missing transformation_analysis field...")
    missing_entries_count = check_transformation_analysis(data)
    
    print(f"📊 Found {missing_entries_count} entries missing transformation_analysis field")
    
    if missing_entries_count > 0:
        # 为缺失字段的条目调用分析器
        print("🧠 Starting analysis for missing entries...")
        analyze_missing_entries(data, analyzer_agent, json_file_path)
        
        # 最后再保存一次，确保所有修改都被写入
        print("💾 Final save...")
        save_json_file(data, json_file_path)
        
        # 验证处理结果
        final_missing = check_transformation_analysis(data)
        print(f"✅ Process completed. Remaining missing entries: {final_missing}")
        print(f"📈 Successfully added transformation_analysis to {missing_entries_count - final_missing} entries")
    else:
        print("✅ All entries already have transformation_analysis field. No changes needed.")
    
    print("🎉 Task completed successfully!")

if __name__ == "__main__":
    main()