import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class StudentService {
  private apiUrl = '/api'; // Proxy ile yönlendirilmiş varsayım

  constructor(private http: HttpClient) {}

  getProfile(): Observable<any> {
    return this.http.get(`${this.apiUrl}/users/me`);
  }

  getLevel(): Observable<any> {
    return this.http.get(`${this.apiUrl}/users/me/level`);
  }

  getSubjects(): Observable<any> {
    return this.http.get(`${this.apiUrl}/subjects/`);
  }

  getStudyPlans(userId: number): Observable<any> {
    return this.http.get(`${this.apiUrl}/study_plans/?user_id=${userId}`);
  }
} 