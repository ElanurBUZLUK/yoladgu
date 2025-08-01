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
  full_name?: string;
  grade?: number;
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
    console.log('AuthService API Config:', ApiConfig.getConfig());
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
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    return this.http.post<LoginResponse>(ApiConfig.getApiUrl('auth/login'), formData)
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

  register(userData: any): Observable<any> {
    return this.http.post(ApiConfig.getApiUrl('auth/register'), userData)
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
    return !!token;
  }

  getToken(): string | null {
    return localStorage.getItem('token');
  }
}