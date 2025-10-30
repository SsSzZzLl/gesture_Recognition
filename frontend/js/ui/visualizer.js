export class Visualizer {
    constructor(overlayId) {
        this.overlay = document.getElementById(overlayId);
        this.ctx = this.overlay.getContext('2d');
    }

    /**
     * 绘制实时手势信息
     * @param {string} action - 手势动作名
     * @param {string} confidence - 置信度(%)
     * @param {number} width - 视频宽度
     * @param {number} height - 视频高度
     */
    drawGestureInfo(action, confidence, width, height) {
        // 清空画布
        this.ctx.clearRect(0, 0, width, height);

        // 绘制信息面板
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(10, 10, 220, 70);

        // 绘制文本
        this.ctx.fillStyle = 'white';
        this.ctx.font = '16px Segoe UI';
        this.ctx.fillText(`动作: ${action}`, 20, 35);
        this.ctx.fillText(`置信度: ${confidence}%`, 20, 60);

        // 绘制置信度进度条
        const barWidth = 200;
        const barHeight = 8;
        const progress = Math.min(100, parseFloat(confidence)) / 100;

        this.ctx.fillStyle = '#eee';
        this.ctx.fillRect(20, 70, barWidth, barHeight);

        this.ctx.fillStyle = progress > 70 ? '#4CAF50' : '#FF9800';
        this.ctx.fillRect(20, 70, barWidth * progress, barHeight);
    }

    /**
     * 调整画布尺寸
     * @param {number} width - 宽度
     * @param {number} height - 高度
     */
    resize(width, height) {
        this.overlay.width = width;
        this.overlay.height = height;
    }
}