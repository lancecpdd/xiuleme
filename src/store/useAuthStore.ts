import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';
import { Invokes, QrLoginUser } from '../components/ui/AppProperties';

export interface AuthUser {
  account: string;
  displayName: string;
  id: string;
  avatarUrl?: string;
}

// checking：启动时正在在线校验已保存的登录态；guest：未登录；authed：已登录
export type AuthStatus = 'checking' | 'guest' | 'authed';

interface AuthState {
  applyQrLogin(qrUser: QrLoginUser): void;
  clearEnteredAsGuest(): void;
  enterAsGuest(): void;
  enteredAsGuest: boolean;
  handleExpired(): void;
  logout(): Promise<void>;
  restoreError: string | null;
  restoreSession(): Promise<void>;
  sessionExpired: boolean;
  status: AuthStatus;
  user: AuthUser | null;
}

const toAuthUser = (qrUser: QrLoginUser): AuthUser => ({
  account: qrUser.userNo || 'paileme',
  displayName: qrUser.nickname || '拍了么用户',
  id: qrUser.userId != null ? String(qrUser.userId) : '',
  avatarUrl: qrUser.avatarUrl || undefined,
});

export const useAuthStore = create<AuthState>((set, get) => ({
  enteredAsGuest: false,
  restoreError: null,
  sessionExpired: false,
  status: 'checking',
  user: null,

  // 游客跳过登录进入（仅内存，不持久化；冷启动或回到空首页会重新看到登录+跳过）
  enterAsGuest: () => {
    set({ enteredAsGuest: true, sessionExpired: false, restoreError: null });
  },

  clearEnteredAsGuest: () => {
    set({ enteredAsGuest: false });
  },

  // 扫码登录成功后写入登录态（token 已由 Rust 端持久化并保存在内存中，不下发给前端）
  applyQrLogin: (qrUser: QrLoginUser) => {
    set({
      user: toAuthUser(qrUser),
      status: 'authed',
      enteredAsGuest: false,
      restoreError: null,
      sessionExpired: false,
    });
  },

  // 启动时由 Rust 端在线校验已保存的登录态；网络异常时按未登录处理，可重试
  restoreSession: async () => {
    set({ status: 'checking', restoreError: null });
    try {
      const saved = await invoke<QrLoginUser | null>(Invokes.QrLoginVerifySaved);
      if (saved && saved.userId != null) {
        set({ user: toAuthUser(saved), status: 'authed', enteredAsGuest: false });
      } else {
        set({ user: null, status: 'guest' });
      }
    } catch (err) {
      console.error('Failed to verify saved auth session:', err);
      set({ user: null, status: 'guest', restoreError: String(err) });
    }
  },

  // 服务端返回 401（token 过期或已吊销）时由 Rust 端通知；保留图库，仅变为 guest
  handleExpired: () => {
    const wasAuthed = get().status === 'authed';
    set({ user: null, status: 'guest', sessionExpired: wasAuthed || get().sessionExpired });
  },

  logout: async () => {
    try {
      await invoke(Invokes.QrLoginLogout);
    } catch (err) {
      console.error('Failed to clear saved auth:', err);
    }
    set({ user: null, status: 'guest', restoreError: null, sessionExpired: false });
  },
}));
