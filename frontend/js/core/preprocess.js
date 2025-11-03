export const Preprocess = {
  /**
   * 预处理图像（适配模型输入要求）
   * @param {HTMLCanvasElement} frame - 原始视频帧
   * @returns {Object} 处理后的张量数据和形状
   */
  process(frame) {
    const targetSize = [128, 128];  // 与模型输入尺寸一致
    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = targetSize[0];
    resizedCanvas.height = targetSize[1];
    const ctx = resizedCanvas.getContext('2d');
    ctx.drawImage(frame, 0, 0, targetSize[0], targetSize[1]);  // 缩放图像

    // 获取像素数据并归一化
    const imageData = ctx.getImageData(0, 0, targetSize[0], targetSize[1]);
    const data = new Float32Array(targetSize[0] * targetSize[1] * 3);  // RGB通道
    const mean = [0.485, 0.456, 0.406];  // ImageNet均值
    const std = [0.229, 0.224, 0.225];   // ImageNet标准差
    let idx = 0;

    // 转换为[C, H, W]格式并标准化
    for (let i = 0; i < imageData.data.length; i += 4) {
      data[idx] = (imageData.data[i] / 255 - mean[0]) / std[0];     // R通道
      data[idx + 1] = (imageData.data[i + 1] / 255 - mean[1]) / std[1]; // G通道
      data[idx + 2] = (imageData.data[i + 2] / 255 - mean[2]) / std[2]; // B通道
      idx += 3;
    }

    return {
      data: data,
      shape: [1, 3, targetSize[0], targetSize[1]]  // [batch, channel, height, width]
    };
  }
};