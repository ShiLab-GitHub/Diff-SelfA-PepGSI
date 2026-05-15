import pandas as pd
from sklearn.model_selection import train_test_split

# 标签映射字典
label_mapping = {
    "fiber": 0,
    "micelle": 1,
    "vesicle": 2,
}

# 读取数据
file_path = "./data/data_all_seq_and_label.csv"  # 替换为你的 CSV 文件路径
df = pd.read_csv(file_path)

# 将标签列的字符串转换为数字
df["labels"] = df["labels"].map(label_mapping)

# 设定交叉验证次数
num_folds = 5

for fold in range(1, num_folds + 1):
    # 以 80%:20% 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=fold * 42)

    # 保存训练集和测试集到 CSV
    train_file = f"multiclass_train_fold_{fold}.csv"
    test_file = f"multiclass_test_fold_{fold}.csv"

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Fold {fold}: 训练集 -> {train_file}, 测试集 -> {test_file}")
