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
  ) {}

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

  updateProgress(progressData: any): Observable<any> {
    return this.http.put(ApiConfig.getApiUrl('users/me/progress'), progressData)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  // Subject specific questions
  getQuestionsBySubject(subjectId: number, limit: number = 10): Observable<any> {
    return this.http.get(ApiConfig.getApiUrl(`questions/?subject_id=${subjectId}&limit=${limit}`))
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  // Topic specific questions
  getQuestionsByTopic(topicId: number, limit: number = 10): Observable<any> {
    return this.http.get(ApiConfig.getApiUrl(`questions/?topic_id=${topicId}&limit=${limit}`))
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }
} 