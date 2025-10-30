export class Webcam {
    constructor(videoId) {
        this.video = document.getElementById(videoId);
        this.isActive = false;
        this.stream = null;
    }

    async start(resolution = { width: 640, height: 480 }, fps = 15) {
        try {
            // 请求摄像头权限
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: resolution.width },
                    height: { ideal: resolution.height },
                    frameRate: { ideal: fps }
                }
            });
            this.video.srcObject = this.stream;
            this.isActive = true;

            // 等待视频加载完成
            return new Promise(resolve => {
                this.video.onloadedmetadata = () => resolve(true);
            });
        } catch (err) {
            console.error("摄像头启动失败：", err);
            alert("请允许摄像头权限后重试");
            return false;
        }
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
            this.isActive = false;
            this.stream = null;
        }
    }

    captureFrame() {
        if (!this.isActive) return null;

        // 创建临时canvas捕获帧
        const canvas = document.createElement('canvas');
        canvas.width = this.video.videoWidth;
        canvas.height = this.video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0, canvas.width, canvas.height);
        return canvas;
    }

    getResolution() {
        return {
            width: this.video.videoWidth,
            height: this.video.videoHeight
        };
    }
}