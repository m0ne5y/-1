from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import sqlite3
import os
import hashlib
# 初始化 Flask 应用
app = Flask(__name__, static_folder="../static", static_url_path="/static")
CORS(app)  # 允许跨域

# 数据库路径
DATABASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../database/recommendation_system.db'))


# 获取数据库连接
def get_db_connection():
    try:
        conn = sqlite3.connect(DATABASE, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

# 根路径路由：直接加载 HTML 页面
@app.route('/', methods=['GET'])
def home():
    return send_from_directory('../templates', 'web.html')


# 登录页面
@app.route('/login')
def login():
    return render_template('login.html')

# 注册页面
@app.route('/register')
def register():
    return render_template('register.html')

# 获取游戏列表
@app.route('/api/games', methods=['GET'])
def get_games():
    try:
        # 获取分页和搜索参数
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        search = request.args.get('search', '')

        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = conn.cursor()
        query = """
            SELECT * FROM games
            WHERE name LIKE ?
            LIMIT ? OFFSET ?
        """
        like_pattern = f"%{search}%"
        offset = (page - 1) * per_page

        cursor.execute(query, (like_pattern, per_page, offset))
        games = cursor.fetchall()

        total_query = "SELECT COUNT(*) FROM games WHERE name LIKE ?"
        total_count = cursor.execute(total_query, (like_pattern,)).fetchone()[0]

        conn.close()

        return jsonify({
            "games": [dict(game) for game in games],
            "total": total_count,
            "page": page,
            "per_page": per_page
        })
    except Exception as e:
        print(f"Error in /api/games: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

# 获取单个游戏详情
@app.route('/api/games/<int:game_id>', methods=['GET'])
def get_game_details(game_id):
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = conn.cursor()
        query = "SELECT * FROM games WHERE id = ?"
        cursor.execute(query, (game_id,))
        game = cursor.fetchone()

        if game:
            conn.close()
            return jsonify(dict(game))
        else:
            conn.close()
            return jsonify({"error": "Game not found"}), 404
    except Exception as e:
        print(f"Error in /api/games/<id>: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


# 推荐接口
@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_id = data.get('user_id')
        item_ids = data.get('item_ids', [])
        item_categories = data.get('item_categories', [])
        hours = data.get('hours', [])

        if not user_id or not item_ids or not item_categories or not hours:
            return jsonify({"error": "Missing required parameters"}), 400

        # 静态占位推荐结果（模型未加载时使用）
        recommended_items = [{"item_id": item_id, "score": 0.5} for item_id in item_ids]
        return jsonify(recommended_items)
    except Exception as e:
        print(f"Error in /api/recommend: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

# 登录接口
@app.route('/api/login', methods=['POST'])
def login_user():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({"error": "用户名或密码不能为空"}), 400

        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = conn.cursor()
        query = "SELECT id, password FROM users WHERE username = ?"
        cursor.execute(query, (username,))
        user = cursor.fetchone()

        if user is None or user['password'] != password:
            return jsonify({"error": "用户名或密码错误"}), 401

        return jsonify({"user_id": user['id']})
    except Exception as e:
        print(f"Error in /api/login: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

# 注册接口
@app.route('/api/register', methods=['POST'])
def register_user():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({"error": "用户名或密码不能为空"}), 400

        # 生成密码哈希
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = conn.cursor()
        # 检查用户名是否已存在
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return jsonify({"error": "用户名已存在"}), 400

        # 插入新用户
        query = "INSERT INTO users (username, password_hash) VALUES (?, ?)"
        cursor.execute(query, (username, password_hash))
        conn.commit()
        conn.close()

        return jsonify({"message": "注册成功"}), 201
    except Exception as e:
        print(f"Error in /api/register: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    print(f"Database path: {DATABASE}")
    app.run(debug=True, port=5000)