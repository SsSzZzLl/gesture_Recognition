import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.es6.min.js';

export const ONNXInfer = {
  session: null,  // 推理会话

  /**
   * 初始化模型
   * @param {string} modelPath - 模型文件路径
   * @returns {Promise<boolean>} 初始化是否成功
   */
  async init(modelPath) {
    try {
      console.log(`🚀 正在加载 ONNX 模型： ${modelPath}`);
      // 检查模型文件是否存在
      const response = await fetch(modelPath, { method: 'HEAD' });
      if (response.ok) {
        const fileSize = parseInt(response.headers.get('content-length') || 0);
        console.log(`✅ 模型文件已找到，大小约：${(fileSize / (1024 * 1024)).toFixed(2)} MB`);
      } else {
        throw new Error(`模型文件不存在或无法访问：${modelPath}`);
      }

      // 创建推理会话
      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],  // 使用WebAssembly后端
        graphOptimizationLevel: 'all'  // 启用所有图优化
      });
      console.log('✅ 模型加载成功（适配onnxruntime-web@1.23.0）');
      return true;
    } catch (err) {
      console.error('❌ 模型加载失败：详细信息');
      console.error('错误类型：', typeof err);
      console.error('错误内容：', err);
      throw new Error(`模型路径或版本不兼容：${modelPath}`);
    }
  },

  /**
   * 执行推理
   * @param {Object} inputTensor - 输入张量
   * @returns {Promise<Object>} 推理结果（标签和置信度）
   */
  async predict(inputTensor) {
    if (!this.session) {
      throw new Error('模型未初始化，请先调用init()');
    }

    try {
      // 创建ONNX张量
      const input = new ort.Tensor('float32', inputTensor.data, inputTensor.shape);
      // 执行推理
      const outputs = await this.session.run({ input: input });
      // 解析结果
      const scores = outputs.output.data;
      const maxIndex = scores.indexOf(Math.max(...scores));
      const labelMap = [
        'move', 'leftclick', 'rightclick',
        'back', 'scrollup', 'scrolldown'
      ];
      return {
        label: labelMap[maxIndex] || '未知',
        confidence: Math.round(scores[maxIndex] * 100)
      };
    } catch (err) {
      console.error('❌ 推理失败：', err);
      throw new Error(`推理过程出错：${err.message}`);
    }
  }
};