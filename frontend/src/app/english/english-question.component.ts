import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../services/api.service';

@Component({
  selector: 'app-english-question',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './english-question.component.html',
  styleUrls: ['./english-question.component.css']
})
export class EnglishQuestionComponent implements OnInit {
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
    
    this.api.getEnglishQuestion().subscribe({
      next: (res) => {
        this.question = res;
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
