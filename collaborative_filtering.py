import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
import joblib


# 加载模拟的用户-游戏交互数据集
# 假设数据文件名为 "simulated_user_game_interactions.csv"
data_path = "simulated_user_game_interactions.csv"
ratings_df = pd.read_csv(data_path)

# 打印数据集信息
print(f"数据集包含 {len(ratings_df)} 条记录")
print(ratings_df.head())

# 使用 surprise 库的 Reader 读取数据
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['UserID', 'GameID', 'Rating']], reader)

# 将数据集分为训练集和测试集
trainset, testset = train_test_split(data, test_size=0.2)

# 使用 KNN 实现基于用户的协同过滤
sim_options = {
    'name': 'cosine',  # 使用余弦相似度
    'user_based': True  # 基于用户的协同过滤
}
algo = KNNBasic(sim_options=sim_options)

# 在训练集上训练模型
algo.fit(trainset)

# 在测试集上生成预测
predictions = algo.test(testset)

# 评估模型性能
rmse = accuracy.rmse(predictions)
print(f"测试集上的 RMSE: {rmse}")
# 在训练完成后添加
MODEL_PATH = "collaborative_filtering_model.pkl"
joblib.dump(algo, MODEL_PATH)
print(f"模型已保存至 {MODEL_PATH}")
# 为一个特定用户生成游戏推荐
def get_top_n_recommendations(algo, user_id, n=10):
    # 获取所有游戏ID
    all_game_ids = ratings_df['GameID'].unique()
    # 获取用户已经评分的游戏
    rated_games = ratings_df[ratings_df['UserID'] == user_id]['GameID'].values
    # 过滤掉用户已经评分的游戏
    unrated_games = [game_id for game_id in all_game_ids if game_id not in rated_games]

    # 为未评分的游戏生成预测评分
    predictions = [algo.predict(user_id, game_id) for game_id in unrated_games]
    # 按评分降序排列并返回前 n 个游戏
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    return [(pred.iid, pred.est) for pred in top_n]

# 示例：为某个用户生成推荐
user_id = "User_123"  # 替换为需要推荐的用户ID
top_recommendations = get_top_n_recommendations(algo, user_id)
print(f"为用户 {user_id} 推荐的游戏：")
for game_id, predicted_rating in top_recommendations:
    print(f"游戏ID: {game_id}, 预测评分: {predicted_rating:.2f}")