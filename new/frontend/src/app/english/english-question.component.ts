import { Component, OnInit } from '@angular/core';
import { ApiService } from '../services/api.service';

@Component({
  selector: 'app-english-question',
  templateUrl: './english-question.component.html',
  styleUrls: ['./english-question.component.css']
})
export class EnglishQuestionComponent implements OnInit {
  question: any;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getEnglishQuestion().subscribe(res => {
      this.question = res;
    });
  }

  submitAnswer(answer: string) {
    console.log("Seçilen cevap:", answer);
    // backend’e cevabı göndermek için eklenebilir
  }
}
