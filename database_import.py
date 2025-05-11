import sqlite3
import pandas as pd
import os

# 数据库路径
DATABASE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'D:\pythonProject4\database\recommendation_system.db'))

# 加载 CSV 文件
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../appstore_games (2).csv'))
data = pd.read_csv(file_path)

# 检查并删除重复的 ID
data = data.drop_duplicates(subset='ID')

# 连接到 SQLite 数据库
conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()

# 清空表
cursor.execute("DELETE FROM games;")
conn.commit()

# 插入数据到 games 表
for _, row in data.iterrows():
    cursor.execute('''
    INSERT OR IGNORE INTO games (id, name, primary_genre, rating, description, icon_url, price, release_date)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        row['ID'],
        row['Name'],
        row['Primary Genre'],
        row['Average User Rating'],
        row['Description'],
        row['Icon URL'],
        row['Price'] if not pd.isna(row['Price']) else None,
        row['Original Release Date']
    ))

# 提交更改并关闭连接
conn.commit()
conn.close()

print("数据已成功插入！")