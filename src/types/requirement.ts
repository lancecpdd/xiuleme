export type RequirementStatus = 'accepted' | 'implementing' | 'rejected' | 'done';

export type RequirementFilter = 'all' | RequirementStatus;

/** 需求池开关（后端 /wx/xiuleme/requirement/info）；功能关闭时 enabled=false */
export interface RequirementInfo {
  enabled: boolean;
}

export interface Requirement {
  id: string;
  title: string;
  content: string;
  status: RequirementStatus | string;
  likeCount: number;
  liked: boolean;
  createTime: string;
  /** 提交者昵称；缺失时前端降级展示 */
  nickname?: string;
  /** 提交者头像 URL；缺失或加载失败时用默认图标 */
  avatarUrl?: string;
}

export interface RequirementLikeResult {
  id: string;
  likeCount: number;
  liked: boolean;
}
