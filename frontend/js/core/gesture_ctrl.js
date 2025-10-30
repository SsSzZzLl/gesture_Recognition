export class GestureController {
    constructor() {
        // 手势-操作映射表
        this.actionMap = {
            "move": this._handleMove.bind(this),
            "leftclick": this._handleLeftClick.bind(this),
            "rightclick": this._handleRightClick.bind(this),
            "back": this._handleBack.bind(this),
            "scrollup": this._handleScrollUp.bind(this),
            "scrolldown": this._handleScrollDown.bind(this)
        };

        // 鼠标移动状态（用于move动作）
        this.lastMousePos = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
        this.moveSensitivity = 5;  // 移动灵敏度
    }

    /**
     * 执行手势对应的操作
     * @param {string} action - 手势动作名
     */
    execute(action) {
        if (this.actionMap[action]) {
            this.actionMap[action]();
        }
    }

    /**
     * 处理move动作（移动鼠标）
     * 简单实现：基于随机偏移模拟（实际需替换为手势定位逻辑）
     */
    _handleMove() {
        // 生成随机偏移（实际项目应基于手势位置计算）
        const dx = (Math.random() - 0.5) * this.moveSensitivity;
        const dy = (Math.random() - 0.5) * this.moveSensitivity;

        // 更新鼠标位置（限制在窗口内）
        this.lastMousePos.x = Math.min(
            window.innerWidth - 10,
            Math.max(10, this.lastMousePos.x + dx)
        );
        this.lastMousePos.y = Math.min(
            window.innerHeight - 10,
            Math.max(10, this.lastMousePos.y + dy)
        );

        // 触发鼠标移动事件
        const event = new MouseEvent('mousemove', {
            clientX: this.lastMousePos.x,
            clientY: this.lastMousePos.y,
            bubbles: true
        });
        document.dispatchEvent(event);
    }

    /**
     * 处理leftclick动作（左键点击）
     */
    _handleLeftClick() {
        const event = new MouseEvent('click', {
            clientX: this.lastMousePos.x,
            clientY: this.lastMousePos.y,
            bubbles: true
        });
        document.dispatchEvent(event);
    }

    /**
     * 处理rightclick动作（右键点击）
     */
    _handleRightClick() {
        const event = new MouseEvent('contextmenu', {
            clientX: this.lastMousePos.x,
            clientY: this.lastMousePos.y,
            bubbles: true
        });
        document.dispatchEvent(event);
    }

    /**
     * 处理back动作（浏览器后退）
     */
    _handleBack() {
        window.history.back();
    }

    /**
     * 处理scrollup动作（页面上滚）
     */
    _handleScrollUp() {
        window.scrollBy(0, -50);
    }

    /**
     * 处理scrolldown动作（页面下滚）
     */
    _handleScrollDown() {
        window.scrollBy(0, 50);
    }
}