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

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.loadQuestion();
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
