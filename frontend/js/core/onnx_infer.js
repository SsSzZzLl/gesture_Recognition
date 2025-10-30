import { preprocess } from './preprocess.js';

export class ONNXInfer {
    constructor() {
        this.session = null;  // ONNX Runtime会话
        this.classes = [
            "move", "leftclick", "rightclick",
            "back", "scrollup", "scrolldown"
        ];  // 与后端类别顺序一致
    }

    /**
     * 加载ONNX模型
     * @param {string} modelPath - 模型路径（如models/base_cnn.onnx）
     * @returns {boolean} 加载成功/失败
     */
    async loadModel(modelPath) {
        try {
            // 配置推理选项（使用CPU）
            const options = {
                executionProviders: ['wasm'],  // 浏览器环境使用wasm
                graphOptimizationLevel: 'all'
            };
            this.session = await ort.InferenceSession.create(modelPath, options);
            return true;
        } catch (err) {
            console.error("模型加载失败：", err);
            return false;
        }
    }

    /**
     * 推理单帧图像
     * @param {HTMLCanvasElement} frameCanvas - 输入帧画布
     * @returns {object} 推理结果（action: 动作名, confidence: 置信度(%), inferTime: 推理耗时(ms)）
     */
    async predict(frameCanvas) {
        if (!this.session) return null;

        try {
            // 预处理
            const { input, preprocessTime } = preprocess(frameCanvas);

            // 构建输入张量
            const tensor = new ort.Tensor('float32', input, [1, 3, 128, 128]);
            const feeds = { input: tensor };

            // 推理
            const start = performance.now();
            const results = await this.session.run(feeds);
            const inferTime = performance.now() - start;

            // 解析结果（取softmax后最大概率）
            const output = results.output.data;
            const softmax = this._softmax(output);
            const maxIdx = softmax.indexOf(Math.max(...softmax));

            return {
                action: this.classes[maxIdx],
                confidence: (softmax[maxIdx] * 100).toFixed(1),
                inferTime,
                preprocessTime
            };
        } catch (err) {
            console.error("推理失败：", err);
            return null;
        }
    }

    /**
     * Softmax激活函数
     * @param {Float32Array} arr - 输入数组
     * @returns {number[]} 归一化后的概率
     */
    _softmax(arr) {
        const exp = arr.map(x => Math.exp(x));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sum);
    }
}