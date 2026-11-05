/** 按点分段数字比较版本号：v1 < v2 返回 -1，相等返回 0，v1 > v2 返回 1 */
export const compareVersions = (v1: string, v2: string) => {
  const parts1 = v1.split('.').map(Number);
  const parts2 = v2.split('.').map(Number);
  const len = Math.max(parts1.length, parts2.length);
  for (let i = 0; i < len; i++) {
    const p1 = parts1[i] || 0;
    const p2 = parts2[i] || 0;
    if (p1 < p2) return -1;
    if (p1 > p2) return 1;
  }
  return 0;
};

/** 前端用户可见的当前版本展示文案 */
export function formatAppVersionForDisplay(version: string) {
  return version;
}
