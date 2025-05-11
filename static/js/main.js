document.addEventListener('DOMContentLoaded', () => {
    // 页面元素引用
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const homeBtn = document.getElementById('home-btn');
    const gameModal = document.getElementById('game-modal');
    const closeModal = document.getElementById('close-modal');
    const recommendBtn = document.getElementById('recommend-btn');
    const recommendationResults = document.getElementById('recommendation-results');
    const recommendationList = document.getElementById('recommendation-list');
    const prevPageBtn = document.getElementById('prev-page-btn');
    const nextPageBtn = document.getElementById('next-page-btn');
    const currentPageElement = document.getElementById('current-page');

    // 用户认证相关元素
    const loginButton = document.getElementById('login-button');
    const registerButton = document.getElementById('register-button');
    const loginModal = document.getElementById('login-modal');
    const registerModal = document.getElementById('register-modal');
    const closeLoginModal = document.getElementById('close-login-modal');
    const closeRegisterModal = document.getElementById('close-register-modal');
    const logoutButton = document.getElementById('logout-button');

    // 状态变量
    let currentPage = 1;
    let currentUserId = localStorage.getItem('currentUserId') || null;

    // 初始化登录状态
    const updateAuthUI = () => {
        const authButtons = document.getElementById('auth-buttons');
        const userAvatar = document.getElementById('user-avatar');

        if (currentUserId) {
            authButtons.classList.add('hidden');
            userAvatar.classList.remove('hidden');
            document.getElementById('username-display').textContent = localStorage.getItem('username');
        } else {
            authButtons.classList.remove('hidden');
            userAvatar.classList.add('hidden');
        }
    };

    // 登录表单提交
    document.getElementById('login-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('login-username').value;
        const password = document.getElementById('login-password').value;

        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            const data = await response.json();

            if (response.ok) {
                currentUserId = data.user_id;
                localStorage.setItem('currentUserId', currentUserId);
                localStorage.setItem('username', username);
                updateAuthUI();
                loginModal.classList.add('hidden');
                showToast(`欢迎回来，${username}！`);
            } else {
                throw new Error(data.error || '登录失败');
            }
        } catch (error) {
            showToast(error.message);
        }
    });

    // 注册表单提交
    document.getElementById('register-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('register-username').value;
        const password = document.getElementById('register-password').value;

        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            if (response.ok) {
                showToast('注册成功，请登录');
                registerModal.classList.add('hidden');
                loginModal.classList.remove('hidden');
            } else {
                const error = await response.json();
                throw new Error(error.error);
            }
        } catch (error) {
            showToast(error.message);
        }
    });

    // 登出功能
    logoutButton.addEventListener('click', () => {
        localStorage.removeItem('currentUserId');
        localStorage.removeItem('username');
        currentUserId = null;
        updateAuthUI();
        showToast('已退出登录');
    });

    // 模态窗口控制
    const modalControls = [
        [loginButton, loginModal, closeLoginModal],
        [registerButton, registerModal, closeRegisterModal]
    ].forEach(([openBtn, modal, closeBtn]) => {
        openBtn.addEventListener('click', () => modal.classList.remove('hidden'));
        closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
        modal.addEventListener('click', (e) => e.target === modal && modal.classList.add('hidden'));
    });

    // 游戏列表加载
    async function loadGames(page = 1) {
        try {
            const search = searchInput.value.trim();
            const response = await fetch(`/api/games?page=${page}&search=${encodeURIComponent(search)}`);

            if (!response.ok) throw new Error('加载失败');

            const data = await response.json();
            renderGameList(data.games);
            updatePagination(data.totalPages, page);
        } catch (error) {
            document.getElementById('game-list').innerHTML = `<p>${error.message}</p>`;
        }
    }

    // 渲染游戏列表
    function renderGameList(games) {
        const gameList = document.getElementById('game-list');
        gameList.innerHTML = games.map(game => `
            <div class="game-item" data-game-id="${game.id}">
                <img src="${game.icon_url}" alt="${game.name}" loading="lazy">
                <div class="game-name">${game.name}</div>
            </div>
        `).join('');

        // 添加点击事件
        document.querySelectorAll('.game-item').forEach(item => {
            item.addEventListener('click', async () => {
                const gameId = item.dataset.gameId;
                try {
                    const game = await fetchGameDetails(gameId);
                    game && showGameDetails(game);
                } catch (error) {
                    showToast('无法加载游戏详情');
                }
            });
        });
    }

    // 获取游戏详情
    async function fetchGameDetails(gameId) {
        const response = await fetch(`/api/games/${gameId}`);
        if (!response.ok) throw new Error('详情加载失败');
        return await response.json();
    }

    // 显示游戏详情模态框
    function showGameDetails(game) {
        ['game-image', 'game-title', 'game-rating', 'game-price', 'game-description'].forEach(id => {
            const element = document.getElementById(id);
            element[id.includes('image') ? 'src' : 'textContent'] =
                game[id.split('-')[1]] || '暂无信息';
        });
        gameModal.classList.remove('hidden');
    }

    // 推荐功能
    recommendBtn.addEventListener('click', async () => {
        if (!currentUserId) {
            showToast('请先登录获取推荐');
            return loginModal.classList.remove('hidden');
        }

        showLoading('正在生成个性化推荐...');

        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: currentUserId })
            });

            if (!response.ok) throw new Error('推荐失败');

            const recommendations = await response.json();
            renderRecommendations(recommendations);
        } catch (error) {
            showToast(error.message);
            recommendationList.innerHTML = `<li>${error.message}</li>`;
        } finally {
            hideLoading();
            recommendationResults.classList.remove('hidden');
        }
    });

    // 渲染推荐结果
    async function renderRecommendations(recommendations) {
        recommendationList.innerHTML = '';

        if (!recommendations.length) {
            recommendationList.innerHTML = '<li>暂无推荐，请尝试更多游戏</li>';
            return;
        }

        // 并行获取游戏详情
        const gameDetails = await Promise.all(
            recommendations.map(item => fetchGameDetails(item.item_id))
        );

        gameDetails.forEach((game, index) => {
            if (!game) return;

            const li = document.createElement('li');
            li.className = 'recommendation-item';
            li.innerHTML = `
                <img src="${game.icon_url}" alt="${game.name}">
                <div class="recommendation-info">
                    <h3>${game.name}</h3>
                    <p>评分预测: ${recommendations[index].score.toFixed(2)}</p>
                    <p>类型: ${game.type || '未知'}</p>
                    <p>点击量: ${game.clicks || 0}</p>
                </div>
            `;
            recommendationList.appendChild(li);
        });
    }

    // 分页控制
    function updatePagination(totalPages, currentPage) {
        prevPageBtn.disabled = currentPage <= 1;
        nextPageBtn.disabled = currentPage >= totalPages;
        currentPageElement.textContent = currentPage;
    }

    // 实用函数
    function showToast(message, duration=3000) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), duration);
    }

    function showLoading(message) {
        const loader = document.getElementById('loading-indicator');
        loader.querySelector('.loading-message').textContent = message;
        loader.classList.remove('hidden');
    }

    function hideLoading() {
        document.getElementById('loading-indicator').classList.add('hidden');
    }

    // 事件监听器
    homeBtn.addEventListener('click', () => {
        searchInput.value = '';
        loadGames();
    });

    searchBtn.addEventListener('click', () => loadGames(1));
    prevPageBtn.addEventListener('click', () => currentPage > 1 && loadGames(--currentPage));
    nextPageBtn.addEventListener('click', () => loadGames(++currentPage));

    // 初始化
    updateAuthUI();
    loadGames();
});