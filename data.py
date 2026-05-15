import os
import random

def generate_data(p=97, train_frac=0.4, out_dir='data'):
    os.makedirs(out_dir, exist_ok=True)
    
    all_data = []
    for x1 in range(p):
        for x2 in range(p):
            x3 = (x1 + x2) % p
            all_data.append((x1, x2, x3))
            
    random.seed(42)
    random.shuffle(all_data)
    
    train_size = int(len(all_data) * train_frac)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]
    
    with open(os.path.join(out_dir, 'train_grok.txt'), 'w') as f:
        for d in train_data:
            f.write(f"{d[0]} {d[1]} {d[2]}\n")
            
    with open(os.path.join(out_dir, 'test_grok.txt'), 'w') as f:
        for d in test_data:
            f.write(f"{d[0]} {d[1]} {d[2]}\n")
            
    print(f"数据生成完毕。")
    print(f"总空间: {len(all_data)}")
    print(f"训练集: {len(train_data)} ({train_frac*100}%)")
    print(f"测试集: {len(test_data)}")

if __name__ == '__main__':
    generate_data()