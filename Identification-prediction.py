import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import set_seed
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch
import numpy as np
from torch.optim import SGD
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter
import datetime
# 初始化 TensorBoard Writer (带时间戳，避免日志覆盖)
from transformers import AutoModel
# --- 配置 ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_checkpoint = "facebook/esm2_t12_35M_UR50D"
model_weight_path = "weight/best_model_fold3_8.4.pth"  # 替换为你的模型权重路径
input_file = "pos_len6.txt"  # 输入文件路径
output_file = "result_len6.txt"  # 输出文件路径

# 类别映射
class_mapping = {0: "fiber", 1: "micelle", 2: "vesicle"}
model_checkpoint1 = "facebook/esm2_t12_35M_UR50D"

# --- 模型定义（需与训练代码一致）---
class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_checkpoint1, num_labels=3000)
        # 2. 解冻部分BERT层（例如最后2层）
        print("Model layers to unfreeze:")
        for name, param in self.bert.named_parameters():
            if 'layer.10' in name or 'layer.11' in name:  # 解冻最后2层
                param.requires_grad = True
                print(f"Unfrozen: {name}")
            else:
                param.requires_grad = False
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(3000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        with torch.no_grad():
            bert_output = self.bert(input_ids=x['input_ids'].to(device), attention_mask=x['attention_mask'].to(device))
        x = self.dropout(bert_output['logits'])
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.output_layer(x)
        return torch.softmax(x, dim=1), x


# --- 加载模型 ---
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = MyModel().to(device)
model.load_state_dict(torch.load(model_weight_path))
model.eval()


# --- 批量预测函数 ---
def predict_batch(sequences):
    inputs = tokenizer(
        sequences,
        max_length=50,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits,_ = model(inputs)
        probs = torch.softmax(logits, dim=1)
        pred_classes = torch.argmax(probs, dim=1).cpu().numpy()
        confidences = torch.max(probs, dim=1).values.cpu().numpy()

    return pred_classes, confidences


# --- 处理输入文件 ---
with open(input_file, "r") as f:
    lines = f.readlines()

# 提取每行的前26个字符作为序列
sequences = [line[:26].strip() for line in lines if line.strip()]

# 批量预测
pred_classes, confidences = predict_batch(sequences)

# --- 写入结果文件 ---
with open(output_file, "w") as f:
    f.write("Sequence\tClass\tConfidence\n")  # 表头
    for seq, cls_id, conf in zip(sequences, pred_classes, confidences):
        f.write(f"{seq}\t{class_mapping[cls_id]}\t{conf:.4f}\n")

print(f"预测完成！结果已保存到 {output_file}")