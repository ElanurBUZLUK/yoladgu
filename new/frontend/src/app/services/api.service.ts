import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private baseUrl = 'http://localhost:8000/api/v1';

  constructor(private http: HttpClient) {}

  getMathQuestion(): Observable<any> {
    return this.http.post(`${this.baseUrl}/math/recommend`, {});
  }

  getEnglishQuestion(): Observable<any> {
    return this.http.post(`${this.baseUrl}/english/questions/generate`, {});
  }
}
