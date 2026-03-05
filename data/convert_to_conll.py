"""
Script để chuyển đổi dữ liệu từ CSV sang định dạng CoNLL cho GNER model
"""
import pandas as pd
import json
import ast
from pathlib import Path
from sklearn.model_selection import train_test_split

def parse_sc_label(sc_label_str):
    """Parse chuỗi sc_label từ CSV thành dict"""
    try:
        # Chuyển string thành dict
        sc_dict = ast.literal_eval(sc_label_str)
        return sc_dict
    except:
        return None

def extract_entity_types(df):
    """Trích xuất tất cả entity types từ dataset"""
    entity_types = set()
    
    for idx, row in df.iterrows():
        sc_label = parse_sc_label(row['sc_label'])
        if sc_label and 'SC' in sc_label:
            for tag in sc_label['SC']:
                if tag != 'O' and tag != 'NULL':
                    # Lấy entity type (bỏ prefix B-/I-)
                    entity_type = tag.split('-', 1)[1] if '-' in tag else tag
                    entity_types.add(entity_type)
    
    return sorted(list(entity_types))

def create_conll_file(df, output_path):
    """Tạo file CoNLL format từ dataframe"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            sc_label = parse_sc_label(row['sc_label'])
            
            if sc_label and 'SC' in sc_label:
                tokens = sc_label.get('text', '').split()
                tags = sc_label.get('SC', [])
                
                # Đảm bảo số lượng tokens và tags khớp nhau
                if len(tokens) == len(tags):
                    for token, tag in zip(tokens, tags):
                        # Bỏ qua các token rỗng
                        if token.strip():
                            f.write(f"{token}\t{tag}\n")
                    # Dòng trống giữa các câu
                    f.write("\n")

def main():
    # Đường dẫn file
    input_file = Path('d:/THUCHANH/models/GNER/data/data_v3.2.csv')
    output_dir = Path('d:/THUCHANH/models/GNER/data/v3.2')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Đọc dữ liệu từ {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Tổng số mẫu: {len(df)}")
    
    # Trích xuất entity types
    print("Trích xuất entity types...")
    entity_types = extract_entity_types(df)
    print(f"Số entity types: {len(entity_types)}")
    print(f"Entity types: {entity_types}")
    
    # Tạo file label.txt
    label_file = output_dir / 'label.txt'
    with open(label_file, 'w', encoding='utf-8') as f:
        for entity_type in entity_types:
            f.write(f"{entity_type}\n")
    print(f"Đã tạo {label_file}")
    
    # Chia dữ liệu: 70% train, 15% dev, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"\nKích thước dữ liệu:")
    print(f"  Train: {len(train_df)} mẫu")
    print(f"  Dev:   {len(dev_df)} mẫu")
    print(f"  Test:  {len(test_df)} mẫu")
    
    # Tạo các file CoNLL
    print("\nTạo file CoNLL...")
    create_conll_file(train_df, output_dir / 'train.txt')
    print(f"  ✓ train.txt")
    
    create_conll_file(dev_df, output_dir / 'dev.txt')
    print(f"  ✓ dev.txt")
    
    create_conll_file(test_df, output_dir / 'test.txt')
    print(f"  ✓ test.txt")
    
    # Tạo thống kê
    print("\n" + "="*60)
    print("THỐNG KÊ ENTITY TYPES:")
    print("="*60)
    
    # Đếm số lượng mỗi entity type
    entity_counts = {}
    for entity_type in entity_types:
        entity_counts[entity_type] = 0
    
    for idx, row in df.iterrows():
        sc_label = parse_sc_label(row['sc_label'])
        if sc_label and 'SC' in sc_label:
            for tag in sc_label['SC']:
                if tag != 'O' and tag != 'NULL' and '-' in tag:
                    entity_type = tag.split('-', 1)[1]
                    if entity_type in entity_counts:
                        entity_counts[entity_type] += 1
    
    for entity_type in sorted(entity_counts.keys(), key=lambda x: entity_counts[x], reverse=True):
        print(f"  {entity_type:20s}: {entity_counts[entity_type]:6d} tokens")
    
    print("\n" + "="*60)
    print(f"Hoàn thành! Dữ liệu đã được lưu tại: {output_dir}")
    print("="*60)
    
    # Hiển thị ví dụ
    print("\nVÍ DỤ CÂU ĐẦU TIÊN TRONG TRAIN.TXT:")
    print("-"*60)
    with open(output_dir / 'train.txt', 'r', encoding='utf-8') as f:
        lines = []
        for line in f:
            if line.strip() == '':
                break
            lines.append(line.strip())
        for line in lines[:10]:  # Hiển thị 10 dòng đầu
            print(line)
        if len(lines) > 10:
            print("...")

if __name__ == '__main__':
    main()
