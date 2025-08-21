import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../services/api.service';
import { EnglishQuestion, EnglishQuestionRequest } from '../models/english-question.model';

@Component({
  selector: 'app-english-question',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './english-question.component.html',
  styleUrls: ['./english-question.component.css']
})
export class EnglishQuestionComponent implements OnInit {
  question: EnglishQuestion | null = null;
  loading = false;
  error: string | null = null;
  
  // Progress tracking
  attemptCount = 0;
  sessionStartTime: Date | null = null;
  isPlacementTest = true; // Will be false after 20 questions
  placementProgress = 0; // 0-20 for placement test

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.initializeSession();
    this.loadQuestion();
  }

  private initializeSession(): void {
    // Check if this is a new session or continuing
    const sessionData = localStorage.getItem('english_session');
    if (sessionData) {
      const session = JSON.parse(sessionData);
      this.attemptCount = session.attemptCount || 0;
      this.sessionStartTime = new Date(session.startTime);
      this.isPlacementTest = session.isPlacementTest !== false;
      this.placementProgress = session.placementProgress || 0;
    } else {
      // New session
      this.sessionStartTime = new Date();
      this.attemptCount = 0;
      this.isPlacementTest = true;
      this.placementProgress = 0;
      this.saveSessionState();
    }
  }

  private saveSessionState(): void {
    const sessionData = {
      attemptCount: this.attemptCount,
      startTime: this.sessionStartTime?.toISOString(),
      isPlacementTest: this.isPlacementTest,
      placementProgress: this.placementProgress
    };
    localStorage.setItem('english_session', JSON.stringify(sessionData));
  }

  loadQuestion(): void {
    this.loading = true;
    this.error = null;
    
    const req: EnglishQuestionRequest = { 
      student_id: 's1', // TODO: Get from auth service
      k: 1 
    };
    
    this.api.getEnglishQuestion(req).subscribe({
      next: (res) => {
        this.question = res.question!;
        this.loading = false;
      },
      error: (error) => {
        console.error('English question error:', error);
        this.error = 'Soru yüklenirken hata oluştu. Lütfen tekrar deneyin.';
        this.loading = false;
      }
    });
  }

  submitAnswer(answer: string) {
    if (!this.question || !this.question.id) {
      console.error('No question available');
      return;
    }

    this.api.submitAnswer(parseInt(this.question.id), answer).subscribe({
      next: (response) => {
        console.log('Answer submitted:', response);
        
        // Update progress tracking
        this.attemptCount++;
        if (this.isPlacementTest) {
          this.placementProgress++;
          if (this.placementProgress >= 20) {
            this.isPlacementTest = false;
            // TODO: Trigger adaptive mode switch
          }
        }
        this.saveSessionState();
        
        // Load next question
        this.loadQuestion();
      },
      error: (error) => {
        console.error('Submit answer error:', error);
        this.error = 'Cevap gönderilirken hata oluştu.';
      }
    });
  }

  getProgressPercentage(): number {
    if (this.isPlacementTest) {
      return (this.placementProgress / 20) * 100;
    }
    return 0; // For adaptive mode, we'll implement different progress tracking
  }

  getSessionDuration(): string {
    if (!this.sessionStartTime) return '0:00';
    
    const now = new Date();
    const diff = now.getTime() - this.sessionStartTime.getTime();
    const minutes = Math.floor(diff / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  retry(): void {
    this.loadQuestion();
  }

  resetSession(): void {
    localStorage.removeItem('english_session');
    this.initializeSession();
    this.loadQuestion();
  }
}
