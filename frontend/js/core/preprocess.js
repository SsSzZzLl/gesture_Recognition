/**
 * 预处理图像帧，与后端训练时的预处理对齐
 * @param {HTMLCanvasElement} frameCanvas - 原始帧画布
 * @returns {Float32Array} 预处理后的张量数据（1x3x128x128）
 */
export function preprocess(frameCanvas) {
    const start = performance.now();

    // 1. 缩放至128x128
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(frameCanvas, 0, 0, 128, 128);

    // 2. 提取像素数据（RGBA）
    const imageData = ctx.getImageData(0, 0, 128, 128).data;

    // 3. 转换为RGB并标准化（与PyTorch transforms对齐）
    const mean = [0.485, 0.456, 0.406];  // ImageNet均值
    const std = [0.229, 0.224, 0.225];   // ImageNet标准差
    const input = new Float32Array(3 * 128 * 128);  // CHW格式

    for (let i = 0; i < 128; i++) {
        for (let j = 0; j < 128; j++) {
            const idx = (i * 128 + j) * 4;  // RGBA索引
            // R通道
            input[i * 128 + j] = (imageData[idx] / 255 - mean[0]) / std[0];
            // G通道
            input[128*128 + i*128 + j] = (imageData[idx + 1] / 255 - mean[1]) / std[1];
            // B通道
            input[2*128*128 + i*128 + j] = (imageData[idx + 2] / 255 - mean[2]) / std[2];
        }
    }

    const preprocessTime = performance.now() - start;
    return { input, preprocessTime };
}