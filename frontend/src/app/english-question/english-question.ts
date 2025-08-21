import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, Router } from '@angular/router';
import { ApiService, Question, QuizResult } from '../services/api';
import englishQuestions from '../../assets/english-questions.json';

@Component({
  selector: 'app-english-question',
  imports: [CommonModule, RouterModule],
  templateUrl: './english-question.html',
  styleUrl: './english-question.scss'
})
export class EnglishQuestionComponent implements OnInit, OnDestroy {
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
    this.apiService.getEnglishQuestions('medium', 5).subscribe({
      next: (response) => {
        console.log('API Response:', response);
        // Try to parse backend response format
        if (response && (response as any).question) {
          // Single question from backend
          const backendQuestion = (response as any).question;
          this.questions = [{
            question: backendQuestion.content || backendQuestion.question,
            options: backendQuestion.options || ["Option A", "Option B", "Option C", "Option D"],
            correctAnswer: this.findCorrectAnswerIndex(backendQuestion.correct_answer, backendQuestion.options),
            explanation: `Difficulty Level: ${backendQuestion.difficulty_level || 'Medium'}`,
            context: backendQuestion.topic_category || "Grammar",
            subject: "english",
            difficulty: "medium"
          }];
        } else if (Array.isArray(response)) {
          // Multiple questions
          this.questions = response.map((q: any) => ({
            question: q.content || q.question,
            options: q.options || ["Option A", "Option B", "Option C", "Option D"],
            correctAnswer: this.findCorrectAnswerIndex(q.correct_answer, q.options),
            explanation: q.explanation || `Difficulty: ${q.difficulty_level || 'Medium'}`,
            context: q.topic_category || "Grammar",
            subject: "english",
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
    const allQuestions = englishQuestions as any[];
    
    // Filter for cloze/fill-in-the-blank questions and convert format
    const clozeQuestions = allQuestions
      .filter(q => q.stem.includes('____') || q.stem.includes('blank') || q.stem.includes('Fill in'))
      .slice(0, 5) // Take first 5 questions
      .map(q => ({
        question: q.stem,
        options: Object.values(q.options) as string[],
        correctAnswer: this.getCorrectAnswerIndex(q.correct_answer, q.options),
        context: `${q.topic} - ${q.subtopic}`,
        explanation: `CEFR Level: ${q.metadata?.cefr_level || 'Unknown'}. Difficulty: ${q.difficulty}`,
        subject: "english",
        difficulty: q.metadata?.cefr_level || "medium"
      }));
    
    // If we don't have enough cloze questions, add some regular questions
    if (clozeQuestions.length < 5) {
      const regularQuestions = allQuestions
        .filter(q => !q.stem.includes('____') && !q.stem.includes('blank'))
        .slice(0, 5 - clozeQuestions.length)
        .map(q => ({
          question: q.stem,
          options: Object.values(q.options) as string[],
          correctAnswer: this.getCorrectAnswerIndex(q.correct_answer, q.options),
          context: `${q.topic} - ${q.subtopic}`,
          explanation: `CEFR Level: ${q.metadata?.cefr_level || 'Unknown'}. Difficulty: ${q.difficulty}`,
          subject: "english",
          difficulty: q.metadata?.cefr_level || "medium"
        }));
      
      this.questions = [...clozeQuestions, ...regularQuestions].slice(0, 5);
    } else {
      this.questions = clozeQuestions;
    }
    
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
    
    // Calculate CEFR level based on performance
    const cefrLevel = this.calculateCEFRLevel(percentage);
    
    // Save progress to backend
    const quizResult: QuizResult = {
      userId: 1, // TODO: Get from auth service
      subject: 'english',
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
        this.showFinalResults(percentage, timeSpent, cefrLevel);
      },
      error: (error) => {
        console.error('Error saving progress:', error);
        this.showFinalResults(percentage, timeSpent, cefrLevel);
      }
    });
  }

  private calculateCEFRLevel(percentage: number): string {
    if (percentage >= 90) return 'C2';
    if (percentage >= 80) return 'C1';
    if (percentage >= 70) return 'B2';
    if (percentage >= 60) return 'B1';
    if (percentage >= 50) return 'A2';
    return 'A1';
  }

  private showFinalResults(percentage: number, timeSpent: number, cefrLevel: string) {
    const minutes = Math.floor(timeSpent / 60);
    const seconds = timeSpent % 60;
    const timeString = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    
    const message = `
🎉 İngilizce Testi Tamamlandı!

📊 Sonuçlar:
✅ Doğru: ${this.score}/${this.questions.length}
📈 Başarı Oranı: %${percentage.toFixed(0)}
⏱️ Süre: ${timeString}
🌍 CEFR Seviyesi: ${cefrLevel}

${this.getCEFRDescription(cefrLevel)}

${this.getRecommendations(percentage)}
    `.trim();

    alert(message);
    this.goHome();
  }

  private getCEFRDescription(level: string): string {
    const descriptions: { [key: string]: string } = {
      'A1': '🔰 Başlangıç seviyesi - Temel kelimeler ve basit cümleler',
      'A2': '📚 Temel seviye - Günlük konuşmalar ve basit metinler',
      'B1': '💬 Orta seviye - Bağımsız iletişim kurabilme',
      'B2': '🎯 İyi seviye - Karmaşık konuları anlayabilme',
      'C1': '🌟 İleri seviye - Akıcı ve doğal iletişim',
      'C2': '🏆 Uzman seviye - Ana dil seviyesinde İngilizce'
    };
    return descriptions[level] || 'Seviye belirlenemedi';
  }

  private getRecommendations(percentage: number): string {
    if (percentage >= 80) {
      return '🎯 Öneriler: Daha zorlu metinler okuyun, native speakerlarla pratik yapın!';
    } else if (percentage >= 60) {
      return '📖 Öneriler: Grammar kurallarını tekrar edin, daha fazla pratik yapın!';
    } else {
      return '📚 Öneriler: Temel grammar kurallarını öğrenin, basit metinler okuyun!';
    }
  }

  goHome() {
    // Navigate to dashboard after quiz completion
    this.router.navigate(['/']);
  }

  // Audio functionality removed - focusing on cloze questions only

  private findCorrectAnswerIndex(correctAnswer: string, options: string[]): number {
    if (!options || !correctAnswer) return 0;
    const index = options.findIndex(option => option === correctAnswer);
    return index >= 0 ? index : 0;
  }

  getOptionLetter(index: number): string {
    return String.fromCharCode(65 + index); // A, B, C, D...
  }
}
