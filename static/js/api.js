const API_BASE_URL = 'http://localhost:5000/api';

// 超时时间（单位：毫秒）
const REQUEST_TIMEOUT = 10000;

// 创建一个带超时的 fetch 请求
async function fetchWithTimeout(resource, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
    try {
        const response = await fetch(resource, { ...options, signal: controller.signal });
        clearTimeout(timeoutId);
        return response;
    } catch (error) {
        clearTimeout(timeoutId);
        throw error; // 抛出超时或其他错误
    }
}

// 获取游戏列表
async function fetchGames(page = 1, search = '') {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/games?page=${page}&search=${encodeURIComponent(search)}`);
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        const data = await response.json();

        // 验证数据结构是否符合预期
        if (!data || !Array.isArray(data.games) || typeof data.totalPages !== 'number') {
            throw new Error("Invalid API response format");
        }

        return data; // 返回游戏列表和总页数
    } catch (error) {
        console.error("Error fetching games:", error);
        return null; // 返回 null 表示失败
    }
}

// 获取单个游戏详情
async function fetchGameDetails(gameId) {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/games/${gameId}`);
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        const data = await response.json();

        // 验证数据结构是否符合预期
        if (!data || typeof data.id !== 'number' || typeof data.name !== 'string') {
            throw new Error("Invalid API response format");
        }

        return data; // 返回游戏详情
    } catch (error) {
        console.error("Error fetching game details:", error);
        return null; // 返回 null 表示失败
    }
}