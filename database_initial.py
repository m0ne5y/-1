import sqlite3
import os

# 数据库路径
DATABASE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'D:\pythonProject4\database\recommendation_system.db'))

# 创建数据库连接
conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()

# 创建 games 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    primary_genre TEXT,
    rating REAL,
    description TEXT,
    icon_url TEXT,
    price REAL,
    release_date TEXT
)
''')

# 创建 users 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)
''')

# 创建 user_behavior 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS user_behavior (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    rating REAL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (game_id) REFERENCES games (id)
)
''')

# 提交更改并关闭连接
conn.commit()
conn.close()

print("数据库初始化完成！")