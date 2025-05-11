import sqlite3
import os

DATABASE = os.path.join(os.path.dirname(__file__), '../database/recommendation_system.db')

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # 创建 games 表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT,
            clicks INTEGER DEFAULT 0,
            added_time TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 创建用户行为表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_behavior (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            game_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()