import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard.component';
import { MathQuestionComponent } from './math/math-question.component';
import { EnglishQuestionComponent } from './english/english-question.component';

const routes: Routes = [
  { path: '', component: DashboardComponent },
  { path: 'math', component: MathQuestionComponent },
  { path: 'english', component: EnglishQuestionComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}
