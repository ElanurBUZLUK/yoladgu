import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';

import { ApiService, MathQuestionResponse, AttemptRequest } from '../services/api';
import { AuthService } from '../services/auth.service';

@Component({
  selector: 'app-math-question',
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './math-question.html',
  styleUrl: './math-question.scss'
})
export class MathQuestionComponent implements OnInit, OnDestroy {
  currentQuestion: MathQuestionResponse | null = null;
  loadingQuestion = false;
  submittingAnswer = false;
  error = '';
  feedback = '';
  selectedChoiceIndex: number | null = null;
  freeFormAnswer = '';
  questionStartTime = 0;
  private nextQuestionTimer: ReturnType<typeof setTimeout> | null = null;
  userId: string | null = null;

  constructor(
    private apiService: ApiService,
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    if (!this.authService.isAuthenticated()) {
      this.router.navigate(['/login']);
      return;
    }

    this.userId = this.authService.getUserId();
    this.loadNextQuestion();
  }

  ngOnDestroy(): void {
    if (this.nextQuestionTimer) {
      clearTimeout(this.nextQuestionTimer);
    }
  }

  loadNextQuestion(): void {
    if (this.nextQuestionTimer) {
      clearTimeout(this.nextQuestionTimer);
      this.nextQuestionTimer = null;
    }

    if (!this.userId) {
      this.userId = this.authService.getUserId();
    }

    if (!this.userId) {
      this.router.navigate(['/login']);
      return;
    }

    this.loadingQuestion = true;
    this.error = '';
    this.feedback = '';
    this.selectedChoiceIndex = null;
    this.freeFormAnswer = '';

    this.apiService.getNextMathQuestion(this.userId).subscribe({
      next: (question) => {
        this.currentQuestion = question;
        this.questionStartTime = Date.now();
        this.loadingQuestion = false;
      },
      error: (err) => {
        this.error = this.extractError(err, 'Bir sonraki soru alınamadı.');
        this.loadingQuestion = false;
      }
    });
  }

  selectChoice(index: number): void {
    if (this.submittingAnswer) {
      return;
    }
    this.selectedChoiceIndex = index;
  }

  submitAnswer(): void {
    if (!this.currentQuestion || !this.userId || this.submittingAnswer || !this.canSubmit()) {
      return;
    }

    const answerValue = this.resolveAnswer();
    if (!answerValue) {
      this.feedback = 'Lütfen bir cevap gir.';
      return;
    }

    const correctAnswer = this.currentQuestion.correct_answer;
    const normalizedAnswer = this.normalizeAnswer(answerValue);
    const isCorrect = correctAnswer
      ? normalizedAnswer === this.normalizeAnswer(correctAnswer)
      : false;

    const timeSpent = Math.max(Date.now() - this.questionStartTime, 0);

    const payload: AttemptRequest = {
      user_id: this.userId,
      item_id: this.currentQuestion.item_id,
      answer: answerValue,
      correct: isCorrect,
      time_ms: timeSpent,
      hints_used: 0,
      context: {}
    };

    this.submittingAnswer = true;
    this.apiService.recordAttempt(payload).subscribe({
      next: () => {
        this.feedback = isCorrect
          ? 'Doğru cevap! Yeni soru hazırlanıyor...'
          : correctAnswer
            ? `Yanlış cevap. Doğru: ${correctAnswer}`
            : 'Cevabın kaydedildi.';

        this.submittingAnswer = false;
        this.queueNextQuestion();
      },
      error: (err) => {
        this.error = this.extractError(err, 'Cevap gönderilirken hata oluştu.');
        this.submittingAnswer = false;
      }
    });
  }

  hasChoices(): boolean {
    return !!this.currentQuestion?.choices && this.currentQuestion.choices.length > 0;
  }

  canSubmit(): boolean {
    if (!this.currentQuestion) {
      return false;
    }
    if (this.hasChoices()) {
      return this.selectedChoiceIndex !== null;
    }

    return this.freeFormAnswer.trim().length > 0;
  }

  getChoiceLetter(index: number): string {
    return String.fromCharCode(65 + index);
  }

  logout(): void {
    this.authService.clearSession();
    this.router.navigate(['/login']);
  }

  private resolveAnswer(): string {
    if (this.hasChoices() && this.selectedChoiceIndex !== null && this.currentQuestion?.choices) {
      return this.currentQuestion.choices[this.selectedChoiceIndex];
    }

    return this.freeFormAnswer.trim();
  }

  private queueNextQuestion(): void {
    if (this.nextQuestionTimer) {
      clearTimeout(this.nextQuestionTimer);
    }

    this.nextQuestionTimer = setTimeout(() => this.loadNextQuestion(), 1500);
  }

  private normalizeAnswer(value: string): string {
    return value.trim().toLowerCase();
  }

  private extractError(error: any, fallback: string): string {
    return error?.error?.detail || error?.message || fallback;
  }
}
