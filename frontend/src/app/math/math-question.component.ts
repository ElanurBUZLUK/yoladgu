import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../services/api.service';
import { MathQuestion, MathQuestionRequest } from '../models/math-question.model';

@Component({
  selector: 'app-math-question',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './math-question.component.html',
  styleUrls: ['./math-question.component.css']
})
export class MathQuestionComponent implements OnInit {
  question: MathQuestion | null = null;
  loading = false;
  error: string | null = null;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.loadQuestion();
  }

  loadQuestion(): void {
    this.loading = true;
    this.error = null;
    
    const req: MathQuestionRequest = { 
      user_id: 's1', // TODO: Get from auth service
      k: 1 
    };
    
    this.api.getMathQuestion(req).subscribe({
      next: (res) => {
        this.question = res.question!;
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

    this.api.submitAnswer(parseInt(this.question.id), answer).subscribe({
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

  getOptionLetter(index: number): string {
    return String.fromCharCode(65 + index); // A, B, C, D...
  }

  retry(): void {
    this.loadQuestion();
  }
}
