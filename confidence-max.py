import pandas as pd

# 输入和输出文件路径
input_file = "result_len6.txt"  # 你的预测结果文件
output_file = "top20_per_class_len6.txt"   # 输出文件路径

# 读取预测结果
df = pd.read_csv(input_file, sep='\t')

# 检查列名是否正确
print("检测到的列名:", df.columns.tolist())

# 确保列名匹配（根据实际文件调整）
# 如果你的文件列名不同，请修改这里的列名
df.columns = ['Sequence', 'Class', 'Confidence']

# 按类别分组并获取每个类别置信度前20的序列
top20_per_class = df.groupby('Class').apply(
    lambda x: x.nlargest(20, 'Confidence')
).reset_index(drop=True)

# 保存结果
top20_per_class.to_csv(output_file, sep='\t', index=False)

print(f"每个类别置信度前20的序列已保存到 {output_file}")
print("\n各类别统计:")
print(top20_per_class['Class'].value_counts())