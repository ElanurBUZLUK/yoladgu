import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map, catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';
import { AuthService } from '../core/services/auth.service';
import { ApiConfig } from '../core/config/api.config';

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

  getMathQuestion(): Observable<any> {
    const payload = {
      user_id: this.authService.getCurrentUserValue()?.id || null
    };

    return this.http.post(
      ApiConfig.getApiUrl('math/recommend'), 
      payload,
      { headers: this.getHeaders() }
    ).pipe(
      map((response: any) => {
        // Backend'ten gelen response'u frontend formatına çevir
        if (response.questions && response.questions.length > 0) {
          const question = response.questions[0];
          
          // Options formatını dönüştür (object -> array)
          let optionsArray = [];
          if (question.options) {
            if (Array.isArray(question.options)) {
              // Eski format: array
              optionsArray = question.options;
            } else if (typeof question.options === 'object') {
              // Yeni format: {A: "1", B: "2", C: "3", D: "4"}
              optionsArray = Object.values(question.options);
            }
          }
          
          return {
            id: question.id,
            text: question.content,
            options: optionsArray,
            optionsMap: question.options, // Orijinal options objesi
            correct_answer: question.correct_answer,
            difficulty_level: question.difficulty_level,
            topic: question.topic_category || question.topic,
            hint: question.hint,
            explanation: question.explanation,
            question_type: question.question_type
          };
        }
        return null;
      }),
      catchError(error => {
        console.error('Math question error:', error);
        return throwError(() => error);
      })
    );
  }

  getEnglishQuestion(): Observable<any> {
    const payload = {
      num_recent_errors: 5
    };

    return this.http.post(
      ApiConfig.getApiUrl('english/questions/generate'), 
      payload,
      { headers: this.getHeaders() }
    ).pipe(
      map((response: any) => {
        // Backend'ten gelen response'u frontend formatına çevir
        if (response.success && response.question) {
          const question = response.question;
          return {
            id: question.id,
            text: question.content,
            options: question.options || [],
            correct_answer: question.correct_answer,
            difficulty_level: question.difficulty_level,
            topic: question.topic_category,
            hint: question.hint,
            explanation: question.explanation,
            error_type: response.generation_info?.error_type
          };
        }
        return null;
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
