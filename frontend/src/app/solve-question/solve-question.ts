import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { interval, Subscription } from 'rxjs';

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
  imports: [CommonModule],
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

  constructor(private router: Router) {}

  ngOnInit() {
    this.loadQuestion();
    this.startTimer();
  }

  ngOnDestroy() {
    this.stopTimer();
  }

  loadQuestion() {
    this.isLoading = true;
    
    // Simulate API call
    setTimeout(() => {
      if (this.currentQuestionIndex < this.mockQuestions.length) {
        this.currentQuestion = this.mockQuestions[this.currentQuestionIndex];
        this.resetQuestionState();
      } else {
        this.currentQuestion = null;
      }
      this.isLoading = false;
    }, 500);
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
      alert('Lütfen bir seçenek seçin!');
      return;
    }

    const isCorrect = this.selectedOption === this.currentQuestion.correct_answer;
    
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

  nextQuestion() {
    if (this.currentQuestionIndex >= this.totalQuestions - 1) {
      this.endQuiz();
      return;
    }

    this.currentQuestionIndex++;
    this.loadQuestion();
  }

  updateProgress() {
    const totalAnswered = this.correctAnswers + this.wrongAnswers;
    this.accuracyPercentage = totalAnswered > 0 ? Math.round((this.correctAnswers / totalAnswered) * 100) : 0;
  }

  generateAIFeedback(isCorrect: boolean) {
    if (isCorrect) {
      this.aiFeedbackText = `
        <strong>Tebrikler! ✅</strong><br>
        Bu soruyu doğru çözdün. ${this.currentQuestion?.topic} konusunda iyi durumdasın. 
        Bir sonraki soruda biraz daha zorlu bir problem ile karşılaşacaksın.
      `;
    } else {
      this.aiFeedbackText = `
        <strong>Üzülme! ❌</strong><br>
        Bu soru yanlış oldu ama endişe etme. ${this.currentQuestion?.topic} konusunda biraz daha çalışman gerekiyor.
        Benzer 3 soru daha çözmen öneriliyor.
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
    
    alert(`Quiz tamamlandı!\n\nSonuçlarınız:\nDoğru: ${this.correctAnswers}\nYanlış: ${this.wrongAnswers}\nDoğruluk Oranı: ${finalAccuracy}%\nToplam Süre: ${this.formatTime(this.elapsedTime)}`);
    
    this.router.navigate(['/']);
  }
}
