import pandas as pd
from sklearn.model_selection import train_test_split


# 读取数据
file_path = "./data/data_all.csv"  # 替换为你的 CSV 文件路径
df = pd.read_csv(file_path)



# 设定交叉验证次数
num_folds = 5

for fold in range(1, num_folds + 1):
    # 以 80%:20% 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=fold * 42)

    # 保存训练集和测试集到 CSV
    train_file = f"train_fold1_{fold}.csv"
    test_file = f"test_fold1_{fold}.csv"

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Fold {fold}: 训练集 -> {train_file}, 测试集 -> {test_file}")
