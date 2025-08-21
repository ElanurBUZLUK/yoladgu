import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { ApiService, UserProgress, AIRecommendation } from '../services/api';

@Component({
  selector: 'app-dashboard',
  imports: [CommonModule, RouterModule],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.scss'
})
export class DashboardComponent implements OnInit {
  userProgress: UserProgress[] = [];
  aiRecommendations: AIRecommendation[] = [];
  loading = true;
  error = '';

  constructor(private apiService: ApiService) {}

  ngOnInit() {
    this.loadDashboardData();
  }

  loadDashboardData() {
    this.loading = true;
    const userId = 1; // TODO: Get from auth service

    // Load user progress
    this.apiService.getUserProgress(userId).subscribe({
      next: (progress) => {
        this.userProgress = progress;
        this.loading = false;
      },
      error: (error) => {
        console.error('Error loading progress:', error);
        this.loading = false;
      }
    });

    // Load AI recommendations
    this.apiService.getAIRecommendations(userId).subscribe({
      next: (recommendations) => {
        this.aiRecommendations = recommendations;
      },
      error: (error) => {
        console.error('Error loading AI recommendations:', error);
        // Fallback recommendations
        this.aiRecommendations = [
          {
            subject: 'Matematik',
            recommendation: 'Trigonometri konusunda zayıfsın. 15 kolay soru çöz.',
            priority: 'high',
            estimatedTime: 30,
            targetScore: 70
          },
          {
            subject: 'İngilizce',
            recommendation: 'Present Perfect Tense konusunu tekrar et.',
            priority: 'medium',
            estimatedTime: 25,
            targetScore: 85
          }
        ];
      }
    });

    // Progress bar animations
    setTimeout(() => {
      const progressBars = document.querySelectorAll('.progress-fill');
      progressBars.forEach((bar: any, index) => {
        const width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => {
          bar.style.width = width;
        }, 300 + (index * 200));
      });
    }, 500);
  }

  getProgressForSubject(subject: string): UserProgress | undefined {
    return this.userProgress.find(p => p.subject.toLowerCase() === subject.toLowerCase());
  }

  getTotalScore(): number {
    if (this.userProgress.length === 0) return 0;
    const totalScore = this.userProgress.reduce((sum, progress) => sum + progress.score, 0);
    return Math.round(totalScore / this.userProgress.length);
  }

  getTotalQuestions(): number {
    return this.userProgress.reduce((sum, progress) => sum + progress.totalQuestions, 0);
  }

  getTotalCorrect(): number {
    return this.userProgress.reduce((sum, progress) => sum + progress.correctAnswers, 0);
  }

  logout() {
    alert('Çıkış yapılıyor...');
    // TODO: Implement logout logic
  }

  openSubject(subject: string) {
    alert(`${subject.charAt(0).toUpperCase() + subject.slice(1)} dersi açılıyor...`);
    // TODO: Navigate to subject page
  }

  startPlan(planTitle: string) {
    alert(`Çalışma planı başlatılıyor: ${planTitle}`);
    // TODO: Start study plan
  }
}
