import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, Router } from '@angular/router';
import { ApiService, Question, QuizResult } from '../services/api';
import mathQuestions from '../../assets/math-questions.json';

@Component({
  selector: 'app-math-question',
  imports: [CommonModule, RouterModule],
  templateUrl: './math-question.html',
  styleUrl: './math-question.scss'
})
export class MathQuestionComponent implements OnInit, OnDestroy {
  questions: Question[] = [];
  currentQuestionIndex = 0;
  currentQuestion: Question | null = null;
  selectedAnswer: number | null = null;
  showResult = false;
  isCorrect = false;
  score = 0;
  timeLeft = 135; // 2:15 in seconds
  timerInterval: any;
  startTime: number = 0;
  loading = true;
  error = '';

  constructor(private apiService: ApiService, private router: Router) {}

  ngOnInit() {
    this.loadQuestions();
  }

  ngOnDestroy() {
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
    }
  }

  loadQuestions() {
    this.loading = true;
    
    // For now, try API first but fallback to static questions immediately if it fails
    this.apiService.getMathQuestions('medium', 5).subscribe({
      next: (response) => {
        console.log('API Response:', response);
        // Try to parse backend response format
        if (response && (response as any).question) {
          // Single question from backend
          const backendQuestion = (response as any).question;
          this.questions = [{
            question: backendQuestion.content || backendQuestion.question,
            options: backendQuestion.options,
            correctAnswer: parseInt(backendQuestion.correct_answer) || backendQuestion.correctAnswer,
            explanation: `Difficulty: ${backendQuestion.difficulty_level}`,
            subject: "math",
            difficulty: "medium"
          }];
        } else if (Array.isArray(response)) {
          // Multiple questions
          this.questions = response.map((q: any) => ({
            question: q.content || q.question,
            options: q.options,
            correctAnswer: q.correct_answer || q.correctAnswer,
            explanation: q.explanation,
            subject: "math",
            difficulty: "medium"
          }));
        } else {
          throw new Error('Invalid response format');
        }
        
        this.loadQuestion();
        this.startTime = Date.now();
        this.loading = false;
      },
      error: (error) => {
        console.error('Error loading questions:', error);
        console.log('Falling back to static questions...');
        this.loadStaticQuestions();
        this.loading = false;
      }
    });
  }

  loadStaticQuestions() {
    // Load questions from the JSON file
    const allQuestions = mathQuestions as any[];
    
    // Convert format and take first 5 questions
    this.questions = allQuestions
      .slice(0, 5)
      .map(q => ({
        question: q.stem,
        options: Object.values(q.options) as string[],
        correctAnswer: this.getCorrectAnswerIndex(q.correct_answer, q.options),
        formula: q.stem, // Use stem as formula for now
        explanation: `Topic: ${q.topic} - ${q.subtopic}. CEFR Level: ${q.metadata?.cefr_level || 'Unknown'}. Difficulty: ${q.difficulty}`,
        subject: "math",
        difficulty: q.metadata?.cefr_level || "medium"
      }));
    
    this.loadQuestion();
    this.startTime = Date.now();
  }

  private getCorrectAnswerIndex(correctAnswer: string, options: any): number {
    const optionKeys = Object.keys(options);
    return optionKeys.indexOf(correctAnswer);
  }

  loadQuestion() {
    if (this.questions.length > 0) {
      this.currentQuestion = this.questions[this.currentQuestionIndex];
      this.selectedAnswer = null;
      this.showResult = false;
      this.timeLeft = 135; // Reset timer for each question
      this.startTimer();
    }
  }

  startTimer() {
    this.timerInterval = setInterval(() => {
      this.timeLeft--;
      if (this.timeLeft <= 0) {
        clearInterval(this.timerInterval);
        this.autoSubmit();
      }
    }, 1000);
  }

  formatTime(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  }

  selectAnswer(index: number) {
    if (!this.showResult) {
      this.selectedAnswer = index;
    }
  }

  submitAnswer() {
    if (this.selectedAnswer === null || this.showResult) return;

    clearInterval(this.timerInterval);
    this.showResult = true;
    this.isCorrect = this.selectedAnswer === this.currentQuestion!.correctAnswer;
    
    if (this.isCorrect) {
      this.score++;
    }
  }

  autoSubmit() {
    if (!this.showResult) {
      this.submitAnswer();
    }
  }

  nextQuestion() {
    if (this.currentQuestionIndex < this.questions.length - 1) {
      this.currentQuestionIndex++;
      this.loadQuestion();
    }
  }

  finishQuiz() {
    const timeSpent = Math.floor((Date.now() - this.startTime) / 1000);
    const percentage = (this.score / this.questions.length) * 100;
    
    // Save progress to backend
    const quizResult: QuizResult = {
      userId: 1, // TODO: Get from auth service
      subject: 'math',
      score: percentage,
      totalQuestions: this.questions.length,
      correctAnswers: this.score,
      timeSpent: timeSpent,
      difficulty: 'medium',
      timestamp: new Date().toISOString()
    };

    this.apiService.saveProgress(quizResult).subscribe({
      next: (response) => {
        console.log('Progress saved:', response);
        alert(`Test tamamlandı!\nDoğru: ${this.score}/${this.questions.length}\nBaşarı oranı: %${percentage.toFixed(0)}\nSüre: ${Math.floor(timeSpent / 60)}:${(timeSpent % 60).toString().padStart(2, '0')}`);
        this.goHome();
      },
      error: (error) => {
        console.error('Error saving progress:', error);
        alert(`Test tamamlandı!\nDoğru: ${this.score}/${this.questions.length}\nBaşarı oranı: %${percentage.toFixed(0)}`);
        this.goHome();
      }
    });
  }

  goHome() {
    // Navigate to dashboard after quiz completion
    this.router.navigate(['/']);
  }

  getOptionLetter(index: number): string {
    return String.fromCharCode(65 + index); // A, B, C, D...
  }
}
