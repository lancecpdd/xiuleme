export type AnnouncementLevel = 'normal' | 'important' | 'urgent';

export const AnnouncementLevels: Record<AnnouncementLevel, AnnouncementLevel> = {
  normal: 'normal',
  important: 'important',
  urgent: 'urgent',
};

export interface Announcement {
  id: string;
  title: string;
  // 富文本 HTML，由拍了么管理后台编辑并存储
  content: string;
  // ISO 8601 时间字符串
  publishedAt: string;
  isPinned?: boolean;
  // 公告等级，缺省按一般处理
  level?: AnnouncementLevel;
}
