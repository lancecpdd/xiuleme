import { invoke } from '@tauri-apps/api/core';
import { Invokes } from '../components/ui/AppProperties';

export type FrontendErrorKind = 'FRONTEND_ERROR' | 'FRONTEND_FATAL';

const MAX_REPORTS_PER_SESSION = 10;
const MAX_MESSAGE_LENGTH = 2000;
const MAX_STACK_LENGTH = 16000;
const IGNORED_MESSAGE_PATTERNS = [/ResizeObserver loop/i];

const reportedKeys = new Set<string>();

interface ErrorDescription {
  message: string;
  stack?: string;
}

function describeError(error: unknown): ErrorDescription {
  if (error instanceof Error) {
    return { message: `${error.name}: ${error.message}`, stack: error.stack };
  }
  if (typeof error === 'string') {
    return { message: error };
  }
  try {
    return { message: JSON.stringify(error) ?? String(error) };
  } catch {
    return { message: String(error) };
  }
}

/**
 * 上报前端未捕获错误（由 Rust 侧落盘并异步上传；debug 构建是否上报由 Rust 侧统一判定）。
 * 会话内按 kind+message+source 去重，且总数有上限；不得调用 console，避免经日志桥递归。
 */
export function reportFrontendError(
  kind: FrontendErrorKind,
  error: unknown,
  source?: string,
  extraStack?: string,
): void {
  if (reportedKeys.size >= MAX_REPORTS_PER_SESSION) {
    return;
  }

  const { message, stack } = describeError(error);
  const trimmedMessage = message.trim().slice(0, MAX_MESSAGE_LENGTH);
  if (!trimmedMessage || IGNORED_MESSAGE_PATTERNS.some((pattern) => pattern.test(trimmedMessage))) {
    return;
  }

  const dedupeKey = `${kind}:${trimmedMessage}:${source ?? ''}`;
  if (reportedKeys.has(dedupeKey)) {
    return;
  }
  reportedKeys.add(dedupeKey);

  const fullStack = [stack, extraStack ? `Component stack:${extraStack}` : undefined]
    .filter(Boolean)
    .join('\n\n')
    .slice(0, MAX_STACK_LENGTH);

  void invoke(Invokes.ReportFrontendError, {
    kind,
    message: trimmedMessage,
    stack: fullStack || null,
    source: source ?? null,
  }).catch(() => {
    // 上报通道不可用时静默丢弃
  });
}
