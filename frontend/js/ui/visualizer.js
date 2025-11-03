export const Visualizer = {
  /**
   * 在视频上绘制识别结果
   * @param {string} videoId - 视频元素ID
   * @param {string} label - 手势标签
   * @param {number} confidence - 置信度
   */
  draw(videoId, label, confidence) {
    const video = document.getElementById(videoId);
    if (!video) return;

    // 创建或获取画布
    let canvas = document.getElementById('visualizer');
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = 'visualizer';
      canvas.style.position = 'absolute';
      canvas.style.top = video.offsetTop + 'px';
      canvas.style.left = video.offsetLeft + 'px';
      canvas.style.pointerEvents = 'none';  // 允许点击穿透
      video.parentNode.appendChild(canvas);
    }

    // 绘制结果
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);  // 清除上一帧

    // 绘制信息面板
    ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
    ctx.fillRect(10, 10, 220, 60);
    ctx.fillStyle = 'black';
    ctx.font = '16px Arial bold';
    ctx.fillText(`手势：${label}`, 20, 35);
    ctx.fillText(`置信度：${confidence}%`, 20, 60);
  }
};