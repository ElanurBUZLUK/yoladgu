import { Component, OnInit } from '@angular/core';
import { ApiService } from '../services/api.service';

@Component({
  selector: 'app-math-question',
  templateUrl: './math-question.component.html',
  styleUrls: ['./math-question.component.css']
})
export class MathQuestionComponent implements OnInit {
  question: any;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getMathQuestion().subscribe(res => {
      this.question = res;
    });
  }

  submitAnswer(answer: string) {
    console.log("Seçilen cevap:", answer);
    // burada backend’e answer submit eklenebilir
  }
}
