import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';
import { ErrorHandlerService } from '../../../core/services/error-handler.service';

@Injectable({ providedIn: 'root' })
export class StudentService {
  private apiUrl = '/api'; // Proxy ile yönlendirilmiş varsayım

  constructor(
    private http: HttpClient,
    private errorHandler: ErrorHandlerService
  ) {}

  getProfile(): Observable<any> {
    return this.http.get(`${this.apiUrl}/users/me`)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  getLevel(): Observable<any> {
    return this.http.get(`${this.apiUrl}/users/me/level`)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  getSubjects(): Observable<any> {
    return this.http.get(`${this.apiUrl}/subjects/`)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  getStudyPlans(userId: number): Observable<any> {
    return this.http.get(`${this.apiUrl}/study_plans/?user_id=${userId}`)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  // Analytics endpoints
  getStudentAnalytics(): Observable<any> {
    return this.http.get(`${this.apiUrl}/analytics/student-analytics`)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  getQuizHistory(limit: number = 10): Observable<any> {
    return this.http.get(`${this.apiUrl}/quiz-sessions?limit=${limit}`)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  getPerformanceStats(): Observable<any> {
    return this.http.get(`${this.apiUrl}/analytics/performance-stats`)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }
} 