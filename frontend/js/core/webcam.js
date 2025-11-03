export const Webcam = {
  /**
   * 初始化摄像头
   * @param {string} videoId - 视频元素ID
   * @returns {Promise<HTMLVideoElement>} 视频元素
   */
  async init(videoId) {
    return new Promise((resolve, reject) => {
      const video = document.getElementById(videoId);
      if (!video) {
        reject(new Error('未找到id为"webcam"的视频元素'));
        return;
      }

      // 请求摄像头权限
      navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
      })
        .then(stream => {
          video.srcObject = stream;
          video.onloadedmetadata = () => {
            video.play();
            console.log('✅ 摄像头初始化成功');
            resolve(video);
          };
        })
        .catch(err => {
          console.error('❌ 摄像头权限请求失败：', err);
          reject(new Error('请允许摄像头权限（浏览器地址栏左侧可设置）'));
        });
    });
  },

  /**
   * 获取当前视频帧
   * @param {string} videoId - 视频元素ID
   * @returns {HTMLCanvasElement|null} 包含当前帧的画布
   */
  getCurrentFrame(videoId) {
    const video = document.getElementById(videoId);
    if (!video || video.readyState !== HTMLMediaElement.HAVE_ENOUGH_DATA) {
      return null;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas;
  }
};