import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, ActivatedRoute } from '@angular/router';
import { interval, Subscription } from 'rxjs';
import { QuestionService } from '../../services/question.service';
import { StudentService } from '../../services/student.service';
import { ErrorHandlerService } from '../../../core/services/error-handler.service';
import { TimerComponent } from '../../../shared/components/timer/timer.component';
import { ProgressBarComponent } from '../../../shared/components/progress-bar/progress-bar.component';
import { DifficultyIndicatorComponent } from '../../../shared/components/difficulty-indicator/difficulty-indicator.component';

interface Question {
  id: number;
  content: string;
  options: string[];
  correct_answer: string;
  difficulty_level: number;
  subject_id: number;
  subject?: string;
  topic?: string;
  hint?: string;
  explanation?: string;
  question_type: string;
  tags?: string[];
  created_by: number;
  is_active: boolean;
}

@Component({
  selector: 'app-solve-question',
  standalone: true,
  imports: [
    CommonModule, 
    TimerComponent, 
    ProgressBarComponent, 
    DifficultyIndicatorComponent
  ],
  templateUrl: './solve-question.html',
  styleUrl: './solve-question.scss'
})
export class SolveQuestionComponent implements OnInit, OnDestroy {
  // Question state
  currentQuestion: Question | null = null;
  currentQuestionIndex = 0;
  totalQuestions = 15;
  selectedOption: string | null = null;
  
  // Progress tracking
  correctAnswers = 0;
  wrongAnswers = 0;
  accuracyPercentage = 0;
  
  // UI state
  showHint = false;
  showExplanation = false;
  showAIFeedback = false;
  showResult = false;
  isLoading = false;
  
  // Timer
  elapsedTime = 0;
  isTimerPulsing = false;
  private timerSubscription?: Subscription;
  
  // AI feedback
  aiFeedbackText = '';
  aiRecommendation = 'Zayıf konulardan başla';
  
  // Mock questions for demo
  private mockQuestions: Question[] = [
    {
      id: 1,
      content: 'Bir üçgende α açısının sinüs değeri 3/5 ise, bu açının kosinüs değeri aşağıdakilerden hangisidir?<br><br><em>(Not: α dar açıdır)</em>',
      options: ['4/5', '3/4', '4/3', '5/4'],
      correct_answer: '4/5',
      difficulty_level: 1,
      subject_id: 1,
      subject: 'Matematik',
      topic: 'Trigonometri',
      hint: 'Pisagor teoremini kullanmayı dene: sin²α + cos²α = 1<br>sin α = 3/5 verildiğine göre, cos²α = 1 - (3/5)² formülünü kullanabilirsin.',
      explanation: '<strong>Doğru cevap: A) 4/5</strong><br><br><strong>Çözüm:</strong><br>sin α = 3/5 verilmiş.<br>Pisagor teoremi: sin²α + cos²α = 1<br>(3/5)² + cos²α = 1<br>9/25 + cos²α = 1<br>cos²α = 1 - 9/25 = 16/25<br>cos α = ±4/5<br>α dar açı olduğu için cos α > 0<br>Dolayısıyla cos α = 4/5',
      question_type: 'multiple_choice',
      tags: ['trigonometri', 'pisagor'],
      created_by: 1,
      is_active: true
    },
    {
      id: 2,
      content: 'x² - 5x + 6 = 0 denkleminin kökleri aşağıdakilerden hangisidir?',
      options: ['x = 2 ve x = 3', 'x = -2 ve x = -3', 'x = 1 ve x = 6', 'x = -1 ve x = -6'],
      correct_answer: 'x = 2 ve x = 3',
      difficulty_level: 1,
      subject_id: 1,
      subject: 'Matematik',
      topic: 'İkinci Derece Denklemler',
      hint: 'Çarpanlara ayırma yöntemini kullan: x² - 5x + 6 = (x - a)(x - b)',
      explanation: '<strong>Doğru cevap: A) x = 2 ve x = 3</strong><br><br><strong>Çözüm:</strong><br>x² - 5x + 6 = 0<br>(x - 2)(x - 3) = 0<br>x - 2 = 0 veya x - 3 = 0<br>x = 2 veya x = 3',
      question_type: 'multiple_choice',
      tags: ['denklem', 'çarpanlara_ayırma'],
      created_by: 1,
      is_active: true
    },
    {
      id: 3,
      content: 'Bir dik üçgende hipotenüs 13 cm, bir dik kenar 5 cm ise, diğer dik kenar kaç cm\'dir?',
      options: ['8 cm', '10 cm', '12 cm', '15 cm'],
      correct_answer: '12 cm',
      difficulty_level: 2,
      subject_id: 1,
      subject: 'Matematik',
      topic: 'Geometri',
      hint: 'Pisagor teoremini kullan: a² + b² = c²',
      explanation: '<strong>Doğru cevap: C) 12 cm</strong><br><br><strong>Çözüm:</strong><br>Pisagor teoremi: a² + b² = c²<br>5² + b² = 13²<br>25 + b² = 169<br>b² = 169 - 25 = 144<br>b = √144 = 12 cm',
      question_type: 'multiple_choice',
      tags: ['geometri', 'pisagor'],
      created_by: 1,
      is_active: true
    }
  ];

  // Filtering parameters
  currentSubjectId?: number;
  currentSubjectName?: string;

  constructor(
    private router: Router,
    private route: ActivatedRoute,
    private questionService: QuestionService,
    private studentService: StudentService,
    private errorHandler: ErrorHandlerService
  ) {}

  ngOnInit() {
    // Check for subject filtering from query params
    this.route.queryParams.subscribe(params => {
      this.currentSubjectId = params['subject_id'] ? parseInt(params['subject_id']) : undefined;
      this.currentSubjectName = params['subject_name'] || undefined;
      
      console.log('Subject filter:', this.currentSubjectName, this.currentSubjectId);
      
      this.loadQuestion();
      this.startTimer();
    });
  }

  ngOnDestroy() {
    this.stopTimer();
  }

  loadQuestion() {
    this.isLoading = true;
    
    // Backend'den gerçek soru al
    this.questionService.getNextQuestion().subscribe({
      next: (question) => {
        this.currentQuestion = question;
        this.resetQuestionState();
        this.isLoading = false;
        console.log('Question loaded:', question);
      },
      error: (error) => {
        console.error('Error loading question:', error);
        this.errorHandler.showWarning('Soru yüklenirken sorun oluştu, demo soru gösteriliyor.');
        // Fallback: Mock data kullan
        this.loadMockQuestion();
      }
    });
  }

  private loadMockQuestion() {
    // Fallback için mock data
    if (this.currentQuestionIndex < this.mockQuestions.length) {
      this.currentQuestion = this.mockQuestions[this.currentQuestionIndex];
      this.resetQuestionState();
    } else {
      this.currentQuestion = null;
    }
    this.isLoading = false;
  }

  resetQuestionState() {
    this.selectedOption = null;
    this.showHint = false;
    this.showExplanation = false;
    this.showAIFeedback = false;
    this.showResult = false;
  }

  selectOption(option: string) {
    if (!this.showResult) {
      this.selectedOption = option;
    }
  }

  showHintSection() {
    this.showHint = true;
  }

  showExplanationSection() {
    this.showExplanation = true;
  }

  checkAnswer() {
    if (!this.selectedOption || !this.currentQuestion) {
      this.errorHandler.showWarning('Lütfen bir seçenek seçin!');
      return;
    }

    this.isLoading = true;
    const startTime = Date.now();
    // Timer başlatıldığından itibaren geçen süre
    const responseTime = this.elapsedTime; // seconds
    
    // Backend'e cevabı ve progress data gönder
    this.questionService.submitAnswer(
      this.currentQuestion.id, 
      this.selectedOption,
      responseTime,
      this.getConfidenceLevel()
    ).subscribe({
      next: (response) => {
        this.handleAnswerResponse(response);
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error submitting answer:', error);
        this.errorHandler.showWarning('Cevap gönderilirken sorun oluştu, yerel değerlendirme yapılıyor.');
        // Fallback: Frontend validation
        this.handleLocalAnswer();
        this.isLoading = false;
      }
    });
  }

  private handleAnswerResponse(response: any) {
    const isCorrect = response.is_correct;
    
    if (isCorrect) {
      this.correctAnswers++;
    } else {
      this.wrongAnswers++;
    }

    this.updateProgress();
    this.updateBackendProgress(); // Backend'e progress gönder
    this.showResult = true;
    this.showExplanation = true;
    this.showAIFeedback = true;
    this.generateAIFeedback(isCorrect, response);
    
    // Backend'den gelen bilgileri kullan
    if (response.message) {
      console.log('Backend message:', response.message);
    }
    
    // Başarı bildirimi
    if (isCorrect) {
      this.errorHandler.showSuccess('Doğru cevap! Tebrikler!');
    }
  }

  private updateBackendProgress() {
    const totalAnswered = this.correctAnswers + this.wrongAnswers;
    const progressData = {
      questions_answered: totalAnswered,
      correct_answers: this.correctAnswers,
      accuracy_rate: totalAnswered > 0 ? (this.correctAnswers / totalAnswered) * 100 : 0,
      total_study_time: this.elapsedTime,
      current_session: {
        session_start: new Date(Date.now() - this.elapsedTime * 1000).toISOString(),
        questions_in_session: totalAnswered,
        session_accuracy: this.accuracyPercentage
      }
    };

    this.questionService.updateProgress(progressData).subscribe({
      next: (response) => {
        console.log('Progress updated:', response);
      },
      error: (error) => {
        console.error('Error updating progress:', error);
        // Error handling zaten QuestionService'te yapılıyor
      }
    });
  }

  private handleLocalAnswer() {
    // Fallback: Local validation
    const isCorrect = this.selectedOption === this.currentQuestion?.correct_answer;
    
    if (isCorrect) {
      this.correctAnswers++;
    } else {
      this.wrongAnswers++;
    }

    this.updateProgress();
    this.showResult = true;
    this.showExplanation = true;
    this.showAIFeedback = true;
    this.generateAIFeedback(isCorrect);
  }

  private getConfidenceLevel(): number {
    // Basit confidence hesaplama (zamanla geliştirilecek)
    const timeTaken = this.elapsedTime;
    if (timeTaken < 30) return 5; // Very confident
    if (timeTaken < 60) return 4; // Confident
    if (timeTaken < 120) return 3; // Neutral
    if (timeTaken < 180) return 2; // Unsure
    return 1; // Very unsure
  }

  nextQuestion() {
    if (this.currentQuestionIndex >= this.totalQuestions - 1) {
      this.endQuiz();
      return;
    }

    this.currentQuestionIndex++;
    this.loadQuestion(); // Yeni soru backend'den gelecek
  }

  updateProgress() {
    const totalAnswered = this.correctAnswers + this.wrongAnswers;
    this.accuracyPercentage = totalAnswered > 0 ? Math.round((this.correctAnswers / totalAnswered) * 100) : 0;
  }

  generateAIFeedback(isCorrect: boolean, response?: any) {
    if (isCorrect) {
      this.aiFeedbackText = `
        <strong>Tebrikler! ✅</strong><br>
        Bu soruyu doğru çözdün. ${this.currentQuestion?.topic} konusunda iyi durumdasın.<br>
        ${response?.points_earned ? `<strong>+${response.points_earned} puan kazandın!</strong><br>` : ''}
        ${response?.current_streak ? `🔥 Seri: ${response.current_streak} doğru!<br>` : ''}
        Bir sonraki soruda biraz daha zorlu bir problem ile karşılaşacaksın.
      `;
    } else {
      this.aiFeedbackText = `
        <strong>Üzülme! ❌</strong><br>
        Bu soru yanlış oldu ama endişe etme. ${this.currentQuestion?.topic} konusunda biraz daha çalışman gerekiyor.<br>
        ${response?.correct_answer ? `<strong>Doğru cevap: ${response.correct_answer}</strong><br>` : ''}
        ${response?.explanation ? `<em>${response.explanation}</em><br>` : ''}
        Benzer sorularla pratik yapman öneriliyor.
      `;
    }
  }

  startTimer() {
    this.timerSubscription = interval(1000).subscribe(() => {
      this.elapsedTime++;
      this.isTimerPulsing = this.elapsedTime % 10 === 0; // Pulse every 10 seconds
    });
  }

  stopTimer() {
    if (this.timerSubscription) {
      this.timerSubscription.unsubscribe();
    }
  }

  formatTime(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  }

  getOptionLetter(index: number): string {
    return String.fromCharCode(65 + index); // A, B, C, D...
  }

  getDifficultyText(level: number): string {
    switch (level) {
      case 1: return 'Kolay';
      case 2: return 'Orta';
      case 3: return 'Zor';
      default: return 'Bilinmiyor';
    }
  }

  getDifficultyClass(level: number): string {
    switch (level) {
      case 1: return 'difficulty-easy';
      case 2: return 'difficulty-medium';
      case 3: return 'difficulty-hard';
      default: return 'difficulty-medium';
    }
  }

  goBack() {
    if (confirm('Quiz\'ten çıkmak istediğiniz emin misiniz? İlerlemeniz kaydedilecek.')) {
      this.router.navigate(['/']);
    }
  }

  endQuiz() {
    this.stopTimer();
    const totalAnswered = this.correctAnswers + this.wrongAnswers;
    const finalAccuracy = totalAnswered > 0 ? Math.round((this.correctAnswers / totalAnswered) * 100) : 0;
    
    // Quiz sonuçlarını backend'e gönder
    this.submitQuizResults({
      total_questions: totalAnswered,
      correct_answers: this.correctAnswers,
      wrong_answers: this.wrongAnswers,
      accuracy_percentage: finalAccuracy,
      total_time_seconds: this.elapsedTime,
      session_data: {
        started_at: new Date(Date.now() - this.elapsedTime * 1000).toISOString(),
        completed_at: new Date().toISOString()
      }
    });
    
    this.errorHandler.showSuccess(
      `Quiz tamamlandı! Doğru: ${this.correctAnswers}, Yanlış: ${this.wrongAnswers}, Oran: ${finalAccuracy}%`
    );
    
    this.router.navigate(['/']);
  }

  private submitQuizResults(results: any) {
    console.log('Quiz results to submit:', results);
    
    // Backend'e quiz sonuçlarını gönder
    this.questionService.submitQuizResults(results).subscribe({
      next: (response) => {
        console.log('Quiz results submitted successfully:', response);
        this.errorHandler.showSuccess('Quiz sonuçları kaydedildi!');
        
        // Progress update de gönder
        this.updateUserProgress();
      },
      error: (error) => {
        console.error('Error submitting quiz results:', error);
        // Error handling zaten QuestionService'te yapılıyor
      }
    });
  }

  private updateUserProgress() {
    const totalAnswered = this.correctAnswers + this.wrongAnswers;
    const accuracy = totalAnswered > 0 ? (this.correctAnswers / totalAnswered) : 0;
    
    const progressData = {
      total_questions_answered: totalAnswered,
      total_correct_answers: this.correctAnswers,
      average_response_time: this.elapsedTime / totalAnswered || 0
    };

    // Inject StudentService for progress update
    import('../features/student/services/student.service').then(module => {
      const studentService = new module.StudentService(
        // Bu services inject edilmeli, şimdilik sadış fonksiyonalite gösterimi
      );
      
      // Note: Bu ideal implementation değil, service'i constructor'da inject etmek gerekiyor
      console.log('Progress data to update:', progressData);
    });
  }
}
