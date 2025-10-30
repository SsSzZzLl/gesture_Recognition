export class DebugPanel {
    constructor() {
        this.fpsEl = document.getElementById('fps');
        this.inferTimeEl = document.getElementById('infer-time');
        this.preprocessTimeEl = document.getElementById('preprocess-time');

        // FPS计算变量
        this.frameCount = 0;
        this.lastFpsTime = performance.now();
    }

    /**
     * 更新帧率显示
     */
    updateFPS() {
        this.frameCount++;
        const now = performance.now();
        const elapsed = now - this.lastFpsTime;

        if (elapsed >= 1000) {  // 每秒更新一次
            const fps = (this.frameCount / (elapsed / 1000)).toFixed(1);
            this.fpsEl.textContent = fps;
            this.frameCount = 0;
            this.lastFpsTime = now;
        }
    }

    /**
     * 更新推理耗时显示
     * @param {number} inferTime - 推理耗时(ms)
     * @param {number} preprocessTime - 预处理耗时(ms)
     */
    updateTimes(inferTime, preprocessTime) {
        this.inferTimeEl.textContent = inferTime.toFixed(1);
        this.preprocessTimeEl.textContent = preprocessTime.toFixed(1);
    }
}