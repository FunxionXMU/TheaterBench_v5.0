import json

# 定义输入和输出文件路径
input_file = '/Users/tongyao/Documents/VSCode code/v5.2/physibench_surprise_v79_unique.json'
output_file = '/Users/tongyao/Documents/VSCode code/v5.2/extracted_middle_captions.json'

def extract_middle_captions():
    try:
        # 读取输入文件
        print(f"正在读取输入文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功读取文件，共包含 {len(data)} 个条目")
        
        # 提取数据
        extracted_data = []
        for i, item in enumerate(data):
            try:
                # 提取关键字
                keyword = item.get('constraints', {}).get('keyword', 'Unknown')
                
                # 提取正常时间线的中间描述
                normal_middle_caption = item.get('normal_timeline', {}).get('middle_caption', '')
                
                # 提取惊喜时间线的中间描述
                surprise_middle_caption = item.get('timeline', {}).get('middle_caption', '')
                
                # 添加到结果列表
                extracted_data.append({
                    'keyword': keyword,
                    'normal_middle_caption': normal_middle_caption,
                    'surprise_middle_caption': surprise_middle_caption
                })
                
                # 打印进度
                if (i + 1) % 10 == 0:
                    print(f"已处理 {i + 1} 个条目")
                    
            except Exception as e:
                print(f"处理第 {i + 1} 个条目时出错: {e}")
        
        # 保存结果到输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n提取完成！共处理 {len(extracted_data)} 个条目")
        print(f"结果已保存到: {output_file}")
        
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_file}'")
    except json.JSONDecodeError:
        print(f"错误：输入文件不是有效的JSON格式")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    print("开始提取middle_caption数据...")
    extract_middle_captions()
