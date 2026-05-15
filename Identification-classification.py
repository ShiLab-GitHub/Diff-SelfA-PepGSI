import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from transformers import AutoTokenizer, AutoModel, set_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import datetime
import warnings

# 过滤 sklearn 在训练初期可能产生的 zero_division警告
warnings.filterwarnings('ignore')

# === 1. 配置参数 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_checkpoint = "facebook/esm2_t12_35M_UR50D"  # 如果显存够大，可用 t33_650M
MAX_LEN = 64  # 针对短序列足够了
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 5e-5  # 微调推荐小学习率
SEED = 42

# 设置随机种子保证复现性
set_seed(SEED)


# === 2. 工具函数 ===
def compute_metrics(y_true, y_pred):
    """一次性计算所有指标"""
    acc = accuracy_score(y_true, y_pred)
    # average='macro' 适用于多分类，关注每一类的表现
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1


# === 3. 数据集定义 ===
class ProteinDataset(Dataset):
    def __init__(self, df):
        self.texts = df["sequence"].tolist()
        self.labels = df["labels"].tolist()

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.texts)


# === 4. 模型定义 (Mean Pooling 优化版) ===
class ESMClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.esm = AutoModel.from_pretrained(model_checkpoint)
        self.hidden_size = self.esm.config.hidden_size

        # --- 解冻策略 ---
        # 冻结大部分参数，只解冻最后 4 层 Encoder 和 Pooler
        for param in self.esm.parameters():
            param.requires_grad = False

        num_layers = self.esm.config.num_hidden_layers
        # 解冻层数：最后4层
        layers_to_unfreeze = [f"layer.{i}" for i in range(num_layers - 4, num_layers)]

        print(f"Unfreezing components: {layers_to_unfreeze}")
        for name, param in self.esm.named_parameters():
            if any(x in name for x in layers_to_unfreeze) or "pooler" in name or "contact_head" in name:
                param.requires_grad = True

        # --- 分类头 ---
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_size, 256)
        self.relu = nn.LeakyReLU()
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state  # (Batch, Seq, Hidden)

        # --- Mean Pooling ---
        # 排除 padding token 的影响
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask  # (Batch, Hidden)

        x = self.dropout(self.relu(self.fc(mean_pooled)))
        logits = self.classifier(x)
        return logits


# === 5. 全局结果记录容器 ===
final_results = {
    'acc': [],
    'prec': [],
    'rec': [],
    'f1': []
}

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# Collate Function for Padding
def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    inputs = tokenizer(texts, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    return inputs, torch.tensor(labels, dtype=torch.long)


# === 6. 训练主循环 ===
for fold in range(5):
    print(f"\n{'=' * 15} Fold {fold + 1} / 5 {'=' * 15}")
    log_dir = f"logs/fold{fold + 1}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    writer = SummaryWriter(log_dir)

    # 读取数据
    df_train = pd.read_csv(f'multiclass_no_none_train_fold_{fold + 1}.csv')
    df_val = pd.read_csv(f'multiclass_no_none_test_fold_{fold + 1}.csv')

    train_loader = DataLoader(ProteinDataset(df_train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ProteinDataset(df_val), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = ESMClassifier(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader),
                                        epochs=EPOCHS)

    # 记录当前 Fold 的最佳指标
    best_val_f1 = -1
    best_metrics_snapshot = {}  # 用于存储 F1 最高时的那一组 acc, prec, rec

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for inputs, labels in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # 计算训练集指标
        t_acc, t_prec, t_rec, t_f1 = compute_metrics(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(inputs['input_ids'], inputs['attention_mask'])
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels.numpy())

        # 计算验证集指标
        v_acc, v_prec, v_rec, v_f1 = compute_metrics(val_labels, val_preds)

        # --- Logging & Printing ---
        print(f"Epoch {epoch + 1:03d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val F1: {v_f1:.4f} (Acc: {v_acc:.4f})")

        # Tensorboard
        writer.add_scalars('Loss', {'train': avg_train_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': t_acc, 'val': v_acc}, epoch)
        writer.add_scalars('Precision', {'train': t_prec, 'val': v_prec}, epoch)
        writer.add_scalars('Recall', {'train': t_rec, 'val': v_rec}, epoch)
        writer.add_scalars('F1', {'train': t_f1, 'val': v_f1}, epoch)

        # --- Save Best Model Logic ---
        # 以 F1-score 作为核心指标来判断模型好坏 (也可以改用 acc)
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            # 这一步很关键：我们记录 F1 最好的这一刻，其他三个指标是多少
            best_metrics_snapshot = {
                'acc': v_acc,
                'prec': v_prec,
                'rec': v_rec,
                'f1': v_f1
            }
            torch.save(model.state_dict(), f"weight/best_model_fold{fold + 1}.pth")
            print(f"    >>> New Best Val F1! Saved. (Prec: {v_prec:.4f}, Rec: {v_rec:.4f})")

    # Fold 结束，将该 Fold 最好的那组结果加入全局记录
    print(f"Fold {fold + 1} Best Results: {best_metrics_snapshot}")
    for k, v in best_metrics_snapshot.items():
        final_results[k].append(v)

    writer.close()

# === 7. 最终结果输出 ===
print("\n" + "=" * 30)
print("Cross-Validation Final Results (Avg of Best Epoch per Fold)")
print("=" * 30)

mean_results = {k: np.mean(v) for k, v in final_results.items()}
std_results = {k: np.std(v) for k, v in final_results.items()}

print(f"Accuracy : {mean_results['acc']:.4f} ± {std_results['acc']:.4f}")
print(f"Precision: {mean_results['prec']:.4f} ± {std_results['prec']:.4f}")
print(f"Recall   : {mean_results['rec']:.4f} ± {std_results['rec']:.4f}")
print(f"F1-Score : {mean_results['f1']:.4f} ± {std_results['f1']:.4f}")
print("=" * 30)