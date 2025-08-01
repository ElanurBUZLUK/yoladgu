import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';
import { ErrorHandlerService } from '../../../core/services/error-handler.service';
import { ApiConfig } from '../../../core/config/api.config';

@Injectable({ providedIn: 'root' })
export class StudentService {
// Remove private apiUrl since we'll use ApiConfig directly

  constructor(
    private http: HttpClient,
    private errorHandler: ErrorHandlerService
  ) {
    console.log('StudentService API Config:', ApiConfig.getConfig());
  }

  getProfile(): Observable<any> {
    return this.http.get(ApiConfig.getApiUrl('users/me'))
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  getLevel(): Observable<any> {
    return this.http.get(ApiConfig.getApiUrl('users/me/level'))
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  getSubjects(): Observable<any> {
    return this.http.get(ApiConfig.getApiUrl('subjects/'))
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  getStudyPlans(userId: number): Observable<any> {
    return this.http.get(ApiConfig.getApiUrl(`study_plans/?user_id=${userId}`))
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  // Analytics endpoints
  getStudentAnalytics(): Observable<any> {
    return this.http.get(ApiConfig.getApiUrl('analytics/student-analytics'))
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  getQuizHistory(limit: number = 10): Observable<any> {
    return this.http.get(ApiConfig.getApiUrl(`quiz-sessions?limit=${limit}`))
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  getPerformanceStats(): Observable<any> {
    return this.http.get(ApiConfig.getApiUrl('analytics/performance-stats'))
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }
} 