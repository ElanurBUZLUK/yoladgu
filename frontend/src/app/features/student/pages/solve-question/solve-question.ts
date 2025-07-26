import { Component, OnInit } from '@angular/core';
import { QuestionService } from '../../services/question.service';

@Component({
  selector: 'app-solve-question',
  templateUrl: './solve-question.html',
  styleUrls: ['./solve-question.scss']
})
export class SolveQuestionComponent implements OnInit {
  question: any = null;
  selectedOption: any = null;
  feedback: string = '';
  showNext: boolean = false;
  loading: boolean = false;

  constructor(private questionService: QuestionService) {}

  ngOnInit(): void {
    this.loadNextQuestion();
  }

  loadNextQuestion() {
    this.loading = true;
    this.feedback = '';
    this.selectedOption = null;
    this.showNext = false;
    this.questionService.getNextQuestion().subscribe(q => {
      this.question = q;
      this.loading = false;
    });
  }

  selectOption(option: any) {
    this.selectedOption = option;
  }

  submitAnswer() {
    if (!this.selectedOption) return;
    this.loading = true;
    this.questionService.submitAnswer(this.question.id, this.selectedOption).subscribe((res: any) => {
      this.feedback = res.correct ? 'Doğru!' : `Yanlış! Doğru cevap: ${res.correct_answer}`;
      if (res.explanation) {
        this.feedback += `\nAçıklama: ${res.explanation}`;
      }
      this.showNext = true;
      this.loading = false;
    });
  }
}
