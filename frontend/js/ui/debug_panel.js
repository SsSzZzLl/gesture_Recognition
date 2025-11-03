export const DebugPanel = {
  /**
   * 更新FPS显示
   * @param {number} fps - 帧率
   */
  updateFPS(fps) {
    const elem = document.getElementById('fps');
    if (elem) elem.textContent = fps.toFixed(1);
  },

  /**
   * 更新置信度显示
   * @param {number} confidence - 置信度（百分比）
   */
  updateConfidence(confidence) {
    const elem = document.getElementById('confidence');
    if (elem) elem.textContent = `${confidence}%`;
  },

  /**
   * 更新手势标签显示
   * @param {string} label - 手势标签
   */
  updateLabel(label) {
    const elem = document.getElementById('gesture-label');
    if (elem) elem.textContent = label;
  }
};