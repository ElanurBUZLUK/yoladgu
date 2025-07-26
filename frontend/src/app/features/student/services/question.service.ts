import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class QuestionService {
  private apiUrl = '/api';

  constructor(private http: HttpClient) {}

  getNextQuestion(): Observable<any> {
    return this.http.get(`${this.apiUrl}/recommendations/next-question`);
  }

  submitAnswer(questionId: number, answer: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/questions/${questionId}/answer`, { answer });
  }
} 