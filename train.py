import pandas as pd
import numpy as np
import paddle
import paddle.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from paddle.io import DataLoader, Dataset
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


# 自定义数据集类
class FLENDataset(Dataset):
    def __init__(self, data):
        self.user_ids = data["UserID"].values
        self.item_ids = data["ItemID"].values
        self.labels = data["Label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        label = self.labels[idx]
        return paddle.to_tensor(user_id, dtype='int64'), paddle.to_tensor(item_id, dtype='int64'), paddle.to_tensor(
            label, dtype='int64')


# 数据增强：平衡数据集
def balance_dataset(data):
    """
    对数据集进行类别平衡，过采样小类别以平衡标签分布。

    参数：
    - data: 原始数据集 DataFrame

    返回：
    - 平衡后的数据集 DataFrame
    """
    grouped = data.groupby("Label")
    max_size = grouped.size().max()
    balanced_data = grouped.apply(lambda x: x.sample(max_size, replace=True)).reset_index(drop=True)
    print(f"平衡后的数据集行数: {len(balanced_data)}")
    return balanced_data


# 改进后的数据生成函数
def generate_data_covering_real(file_path, num_users=1000, num_games_per_user=50, behaviors=["click", "rate", "buy"]):
    """
    改进后的数据生成函数：覆盖所有真实游戏数据，并生成多对多用户-游戏行为数据。

    参数：
    - file_path: 原始数据文件路径
    - num_users: 用户数量
    - num_games_per_user: 每个用户关联的游戏数量
    - behaviors: 用户行为列表

    返回：
    - 生成后的数据集 DataFrame
    - 用户数量
    - 游戏数量
    """
    # 加载原始数据
    original_data = pd.read_csv(file_path)
    num_games = len(original_data)
    game_ids = list(range(num_games))

    # 构建数据集
    generated_data = []

    # 确保覆盖所有游戏
    for game_id in game_ids:
        user_id = np.random.randint(0, num_users)  # 随机分配一个用户
        behavior = np.random.choice(behaviors, p=[0.7, 0.2, 0.1])  # 行为分布
        generated_data.append({
            "UserID": user_id,
            "ItemID": game_id,
            "Behavior": behavior,
            "Label": {"click": 0, "rate": 1, "buy": 2}[behavior]
        })

    # 生成多对多的用户-游戏行为
    for user_id in range(num_users):
        # 随机选择 num_games_per_user 个游戏
        user_game_ids = np.random.choice(game_ids, size=num_games_per_user, replace=False)
        for game_id in user_game_ids:
            behavior = np.random.choice(behaviors, p=[0.7, 0.2, 0.1])  # 行为分布
            generated_data.append({
                "UserID": user_id,
                "ItemID": game_id,
                "Behavior": behavior,
                "Label": {"click": 0, "rate": 1, "buy": 2}[behavior]
            })

    # 转换为 DataFrame
    generated_df = pd.DataFrame(generated_data)

    # 打印范围以验证
    print(f"生成的数据集行数: {len(generated_df)}")
    print(f"UserID 范围: 0 ~ {generated_df['UserID'].max()}")
    print(f"ItemID 范围: 0 ~ {generated_df['ItemID'].max()}")

    return generated_df, num_users, num_games


# 改进后的模型定义
class FLEN(nn.Layer):
    def __init__(self, num_users, num_items, embed_dim=128, num_classes=3):
        super(FLEN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.3)  # Dropout 防止过拟合
        self.relu = nn.ReLU()

    def forward(self, user_id, item_id):
        user_embed = self.user_embedding(user_id)
        item_embed = self.item_embedding(item_id)
        x = paddle.concat([user_embed, item_embed], axis=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        logits = self.fc4(x)
        return logits


# 改进后的训练函数
def train_flen(data, num_users, num_items, embed_dim=128, batch_size=512, epochs=30, learning_rate=0.0001):
    # 数据集划分
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Label"])
    train_dataset = FLENDataset(train_data)
    val_dataset = FLENDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = FLEN(num_users=num_users, num_items=num_items, embed_dim=embed_dim, num_classes=3)
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=learning_rate, weight_decay=0.01)
    class_weights = paddle.to_tensor([1.0, 3.5, 7.0], dtype="float32")  # 类别权重
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for user_id, item_id, label in train_loader:
            logits = model(user_id, item_id)
            loss = loss_fn(logits, label)

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.numpy()

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}")

    print("训练完成！")
    return model, val_loader


# 验证函数
def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with paddle.no_grad():
        for user_id, item_id, label in val_loader:
            logits = model(user_id, item_id)
            pred = paddle.argmax(logits, axis=1)
            correct += (pred == label).sum().numpy()
            total += label.shape[0]
            all_labels.extend(label.numpy())
            all_preds.extend(pred.numpy())
    accuracy = correct / total
    print(f"验证集准确率: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["click", "rate", "buy"]))
    ConfusionMatrixDisplay.from_predictions(all_labels, all_preds, display_labels=["click", "rate", "buy"])
    return accuracy


# 主程序
if __name__ == "__main__":
    # 数据生成
    input_file = "appstore_games (2).csv"  # 替换为您的文件路径
    data, num_users, num_items = generate_data_covering_real(input_file, num_users=1000, num_games_per_user=50)

    # 平衡数据集
    balanced_data = balance_dataset(data)

    # 训练模型
    model, val_loader = train_flen(balanced_data, num_users=num_users, num_items=num_items)

    # 验证模型
    evaluate(model, val_loader)

    # 保存模型
    model_path = "flen_model_optimized.pdparams"
    paddle.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")