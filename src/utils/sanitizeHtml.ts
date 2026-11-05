import DOMPurify from 'dompurify';

// 富文本中的链接统一在外部浏览器打开，避免 WebView 内跳转
DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A') {
    node.setAttribute('target', '_blank');
    node.setAttribute('rel', 'noopener noreferrer');
  }
});

export const sanitizeHtml = (html: string): string => DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
