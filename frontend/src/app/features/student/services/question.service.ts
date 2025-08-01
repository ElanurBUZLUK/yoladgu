import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { throwError } from 'rxjs';
import { ErrorHandlerService } from '../../../core/services/error-handler.service';

interface QuestionResponse {
  id: number;
  content: string;
  options: string[];
  correct_answer: string;
  difficulty_level: number;
  subject_id: number;
  subject?: string;
  topic?: string;
  hint?: string;
  explanation?: string;
  question_type: string;
  tags?: string[];
  created_by: number;
  is_active: boolean;
}

interface SubmitAnswerResponse {
  is_correct: boolean;
  correct_answer?: string;
  explanation?: string;
  points_earned?: number;
  current_streak?: number;
  message?: string;
}

@Injectable({ providedIn: 'root' })
export class QuestionService {
  private apiUrl = '/api/v1'; // Backend API v1 prefix kullanıyor

  constructor(
    private http: HttpClient,
    private errorHandler: ErrorHandlerService
  ) {}

  getNextQuestion(): Observable<QuestionResponse> {
    return this.http.get<QuestionResponse>(`${this.apiUrl}/recommendations/next-question`)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  submitAnswer(questionId: number, answer: string, responseTime?: number, confidenceLevel?: number): Observable<SubmitAnswerResponse> {
    const payload: any = { answer };
    if (responseTime) payload.response_time = responseTime;
    if (confidenceLevel) payload.confidence_level = confidenceLevel;
    
    return this.http.post<SubmitAnswerResponse>(`${this.apiUrl}/questions/${questionId}/answer`, payload)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  // Yeni method: Random question almak için (fallback)
  getRandomQuestion(): Observable<QuestionResponse> {
    return this.http.get<QuestionResponse>(`${this.apiUrl}/questions/random`)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  // Quiz sonuçlarını backend'e gönder
  submitQuizResults(results: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/quiz-sessions`, results)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }

  // Öğrenci progress data güncelle
  updateProgress(progressData: any): Observable<any> {
    return this.http.put(`${this.apiUrl}/users/me/progress`, progressData)
      .pipe(
        catchError(error => {
          this.errorHandler.handleHttpError(error);
          return throwError(() => error);
        })
      );
  }
} 