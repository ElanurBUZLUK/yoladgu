import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map, catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';
import { AuthService } from '../core/services/auth.service';
import { ApiConfig } from '../core/config/api.config';
import { EnglishQuestionRequest, EnglishQuestionResponse } from '../models/english-question.model';
import { MathQuestionRequest, MathQuestionResponse } from '../models/math-question.model';

@Injectable({ providedIn: 'root' })
export class ApiService {
  constructor(
    private http: HttpClient,
    private authService: AuthService
  ) {}

  private getHeaders(): HttpHeaders {
    const token = this.authService.getToken();
    return new HttpHeaders({
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` })
    });
  }

  getMathQuestion(req: MathQuestionRequest): Observable<MathQuestionResponse> {
    return this.http
      .post<MathQuestionResponse>(ApiConfig.getApiUrl('math/questions/generate'), req, { headers: this.getHeaders() })
      .pipe(
        map((res) => {
          if (!res || res.success !== true || !res.question) {
            throw new Error('Invalid response contract');
          }
          // Invariant (runtime guard): 1 doğru + benzersiz distraktörler
          const opts = res.question.options || [];
          if (!opts.includes(res.question.correct_answer)) {
            throw new Error('Answer not in options');
          }
          if (new Set(opts).size !== opts.length) {
            throw new Error('Duplicate options');
          }
          return res;
        }),
        catchError(error => {
          console.error('Math question error:', error);
          return throwError(() => error);
        })
      );
  }

  getEnglishQuestion(req: EnglishQuestionRequest): Observable<EnglishQuestionResponse> {
    return this.http
      .post<EnglishQuestionResponse>(ApiConfig.getApiUrl('english/questions/generate'), req, { headers: this.getHeaders() })
      .pipe(
        map((res) => {
          if (!res || res.success !== true || !res.question) {
            throw new Error('Invalid response contract');
          }
          // Invariant (runtime guard): 1 doğru + benzersiz distraktörler
          const opts = res.question.options || [];
          if (!opts.includes(res.question.correct_answer)) {
            throw new Error('Answer not in options');
          }
          if (new Set(opts).size !== opts.length) {
            throw new Error('Duplicate options');
          }
          return res;
        }),
        catchError(error => {
          console.error('English question error:', error);
          return throwError(() => error);
        })
      );
  }

  submitAnswer(questionId: number, answer: string): Observable<any> {
    const payload = {
      answer: answer,
      question_id: questionId
    };

    return this.http.post(
      ApiConfig.getApiUrl(`questions/${questionId}/answer`),
      payload,
      { headers: this.getHeaders() }
    ).pipe(
      catchError(error => {
        console.error('Submit answer error:', error);
        return throwError(() => error);
      })
    );
  }
}
