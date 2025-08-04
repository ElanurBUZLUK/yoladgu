import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';
import { map, tap, catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';
import { Router } from '@angular/router';
import { ErrorHandlerService } from './error-handler.service';
import { ApiConfig } from '../config/api.config';

interface User {
  id: number;
  username: string;
  email: string;
  full_name: string;  // Backend'te required
  grade?: string;     // Backend'te string olarak tanımlı
  role?: string;      // Backend'te UserRole enum
  is_active: boolean;
}

interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
// Remove private apiUrl since we'll use ApiConfig directly
  private currentUserSubject = new BehaviorSubject<User | null>(null);
  public currentUser$ = this.currentUserSubject.asObservable();

  constructor(
    private http: HttpClient,
    private router: Router,
    private errorHandler: ErrorHandlerService
  ) {
    this.checkStoredToken();
  }

  private checkStoredToken() {
    const token = localStorage.getItem('token');
    if (token) {
      this.getCurrentUser().subscribe({
        next: (user) => this.currentUserSubject.next(user),
        error: () => this.logout()
      });
    }
  }

  login(username: string, password: string): Observable<LoginResponse> {
    const body = new URLSearchParams();
    body.set('username', username);
    body.set('password', password);

    const headers = {
      'Content-Type': 'application/x-www-form-urlencoded'
    };

    return this.http.post<LoginResponse>(ApiConfig.getApiUrl('auth/login'), body.toString(), { headers })
      .pipe(
        tap(response => {
          localStorage.setItem('token', response.access_token);
          this.currentUserSubject.next(response.user);
          this.errorHandler.showSuccess('Giriş başarılı!');
        }),
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  register(userData: any): Observable<User> {
    return this.http.post<User>(ApiConfig.getApiUrl('auth/register'), userData)
      .pipe(
        tap(() => {
          this.errorHandler.showSuccess('Kayıt başarılı! Giriş yapabilirsiniz.');
        }),
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  logout(): void {
    localStorage.removeItem('token');
    this.currentUserSubject.next(null);
    this.errorHandler.showInfo('Çıkış yapıldı.');
    this.router.navigate(['/login']);
  }

  getCurrentUser(): Observable<User> {
    return this.http.get<User>(ApiConfig.getApiUrl('users/me'))
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  isAuthenticated(): boolean {
    const token = localStorage.getItem('token');
    if (!token) {
      return false;
    }

    // Token expiry kontrolü
    if (this.isTokenExpired(token)) {
      this.logout();
      return false;
    }

    return true;
  }

  getToken(): string | null {
    return localStorage.getItem('token');
  }

  private isTokenExpired(token: string): boolean {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      const exp = payload.exp * 1000; // Convert to milliseconds
      return Date.now() >= exp;
    } catch (error) {
      // Token geçersiz ise expired kabul et
      return true;
    }
  }
}
