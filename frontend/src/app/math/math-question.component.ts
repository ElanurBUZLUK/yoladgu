import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../services/api.service';

@Component({
  selector: 'app-math-question',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './math-question.component.html',
  styleUrls: ['./math-question.component.css']
})
export class MathQuestionComponent implements OnInit {
  question: any;
  loading = false;
  error: string | null = null;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.loadQuestion();
  }

  loadQuestion(): void {
    this.loading = true;
    this.error = null;
    
    this.api.getMathQuestion().subscribe({
      next: (res) => {
        this.question = res;
        this.loading = false;
      },
      error: (error) => {
        console.error('Math question error:', error);
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

    this.api.submitAnswer(this.question.id, answer).subscribe({
      next: (response) => {
        console.log('Answer submitted:', response);
        // Cevap gönderildikten sonra yeni soru yükle
        this.loadQuestion();
      },
      error: (error) => {
        console.error('Submit answer error:', error);
        this.error = 'Cevap gönderilirken hata oluştu.';
      }
    });
  }

  retry(): void {
    this.loadQuestion();
  }
}
