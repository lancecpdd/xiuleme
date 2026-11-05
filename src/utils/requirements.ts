import { invoke } from '@tauri-apps/api/core';
import { Invokes } from '../components/ui/AppProperties';
import {
  Requirement,
  RequirementInfo,
  RequirementLikeResult,
  RequirementStatus,
} from '../types/requirement';

// 由 Rust 侧请求拍了么业务接口。已登录时 Rust 会附带 token，用于返回是否已赞。
export const fetchRequirementInfo = async (): Promise<RequirementInfo> =>
  invoke<RequirementInfo>(Invokes.FetchRequirementInfo);

export const fetchRequirements = async (status?: RequirementStatus): Promise<Array<Requirement>> =>
  invoke<Array<Requirement>>(Invokes.FetchRequirements, { status: status ?? null });

export const submitRequirement = async (title: string, content: string): Promise<string> =>
  invoke<string>(Invokes.SubmitRequirement, { title, content });

export const toggleRequirementLike = async (id: string): Promise<RequirementLikeResult> =>
  invoke<RequirementLikeResult>(Invokes.ToggleRequirementLike, { id });
