import { invoke } from '@tauri-apps/api/core';
import { Invokes } from '../components/ui/AppProperties';
import { Announcement } from '../types/announcement';

const sortAnnouncements = (items: Array<Announcement>): Array<Announcement> =>
  [...items].sort((a, b) => {
    if (!!a.isPinned !== !!b.isPinned) {
      return a.isPinned ? -1 : 1;
    }
    return new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime();
  });

// 由 Rust 侧请求拍了么管理后台（app_config 的 adminApiBaseUrl），前端不直连 HTTP。
export const fetchAnnouncements = async (): Promise<Array<Announcement>> => {
  const items = await invoke<Array<Announcement>>(Invokes.FetchAnnouncements);
  return sortAnnouncements(items);
};
