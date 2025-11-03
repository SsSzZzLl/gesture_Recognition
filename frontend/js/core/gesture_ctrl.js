export const GestureCtrl = {
  /**
   * 执行手势对应的操作
   * @param {string} label - 手势标签
   */
  execute: (label) => {
    switch (label) {
      case 'leftclick':
        console.log('🖱️ 执行左键点击');
        // 可添加实际点击逻辑（如使用robotjs等库）
        break;
      case 'rightclick':
        console.log('🖱️ 执行右键点击');
        break;
      case 'scrollup':
        console.log('📜 向上滚动');
        break;
      case 'scrolldown':
        console.log('📜 向下滚动');
        break;
      case 'move':
        console.log('➡️ 移动操作');
        break;
      case 'back':
        console.log('🔙 返回操作');
        break;
      default:
        break;
    }
  }
};