import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Question {
  id?: number;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation?: string;
  context?: string;
  audioUrl?: string;
  formula?: string;
  subject: string;
  difficulty: string;
  category?: string;
}

export interface UserProgress {
  userId: number;
  subject: string;
  totalQuestions: number;
  correctAnswers: number;
  score: number;
  lastAttempted: string;
}

export interface QuizResult {
  userId: number;
  subject: string;
  score: number;
  totalQuestions: number;
  correctAnswers: number;
  timeSpent: number;
  difficulty: string;
  timestamp: string;
}

export interface AIRecommendation {
  subject: string;
  recommendation: string;
  priority: 'high' | 'medium' | 'low';
  estimatedTime: number;
  targetScore: number;
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = 'http://localhost:8000/api/v1'; // Test Backend API URL

  constructor(private http: HttpClient) { }

  // Auth endpoints
  login(email: string, password: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/auth/login`, { email, password });
  }

  register(email: string, password: string, name: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/auth/register`, { email, password, name });
  }

  // Question endpoints
  getQuestions(subject: string, difficulty: string = 'medium', count: number = 5): Observable<Question[]> {
    const params = new HttpParams()
      .set('difficulty', difficulty)
      .set('count', count.toString());
    
    return this.http.get<Question[]>(`${this.baseUrl}/questions/${subject}`, { params });
  }

  getMathQuestions(student_id: number = 1, k: number = 5): Observable<any> {
    // Use the real ML/RAG endpoint for math questions
    return this.http.post<any>(`${this.baseUrl}/math/questions/generate`, {
      user_id: student_id.toString(),
      k: k
    });
  }

  getEnglishQuestions(student_id: number = 1, k: number = 5): Observable<any> {
    // Use the real ML/RAG endpoint for English questions  
    return this.http.post<any>(`${this.baseUrl}/english/questions/generate`, {
      student_id: student_id.toString(),
      k: k
    });
  }

  // Progress tracking
  saveProgress(progress: QuizResult): Observable<any> {
    return this.http.post(`${this.baseUrl}/progress/save`, progress);
  }

  getUserProgress(userId: number, subject?: string): Observable<UserProgress[]> {
    let params = new HttpParams().set('userId', userId.toString());
    if (subject) {
      params = params.set('subject', subject);
    }
    
    return this.http.get<UserProgress[]>(`${this.baseUrl}/progress/user`, { params });
  }

  // AI Recommendations
  getAIRecommendations(userId: number): Observable<AIRecommendation[]> {
    const params = new HttpParams().set('userId', userId.toString());
    return this.http.get<AIRecommendation[]>(`${this.baseUrl}/ai/recommendations`, { params });
  }

  // Dashboard data
  getDashboardData(userId: number): Observable<any> {
    return this.http.get(`${this.baseUrl}/dashboard/progress?student_id=${userId}`);
  }

  getDashboardRecommendations(userId: number): Observable<any> {
    return this.http.get(`${this.baseUrl}/dashboard/recommendations?student_id=${userId}`);
  }

  // Question generation
  generateQuestions(subject: string, topic: string, count: number = 5): Observable<Question[]> {
    return this.http.post<Question[]>(`${this.baseUrl}/questions/generate`, {
      subject,
      topic,
      count
    });
  }

  // RAG system
  searchQuestions(query: string, subject: string): Observable<Question[]> {
    const params = new HttpParams()
      .set('query', query)
      .set('subject', subject);
    
    return this.http.get<Question[]>(`${this.baseUrl}/questions/search`, { params });
  }

  // User management
  getUserProfile(userId: number): Observable<any> {
    return this.http.get(`${this.baseUrl}/users/${userId}`);
  }

  updateUserProfile(userId: number, profile: any): Observable<any> {
    return this.http.put(`${this.baseUrl}/users/${userId}`, profile);
  }

  // Error handling
  private handleError(error: any): Observable<never> {
    console.error('API Error:', error);
    throw error;
  }
}
