import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private accessTokenKey = 'aqs_access_token';
  private refreshTokenKey = 'aqs_refresh_token';
  private userIdKey = 'aqs_user_id';

  setSession(accessToken: string, refreshToken: string | null, userId: string): void {
    if (accessToken) {
      localStorage.setItem(this.accessTokenKey, accessToken);
    }

    if (refreshToken) {
      localStorage.setItem(this.refreshTokenKey, refreshToken);
    }

    localStorage.setItem(this.userIdKey, userId);
  }

  clearSession(): void {
    localStorage.removeItem(this.accessTokenKey);
    localStorage.removeItem(this.refreshTokenKey);
    localStorage.removeItem(this.userIdKey);
  }

  getAccessToken(): string | null {
    return localStorage.getItem(this.accessTokenKey);
  }

  getRefreshToken(): string | null {
    return localStorage.getItem(this.refreshTokenKey);
  }

  getUserId(): string | null {
    return localStorage.getItem(this.userIdKey);
  }

  isAuthenticated(): boolean {
    return !!this.getAccessToken();
  }
}
