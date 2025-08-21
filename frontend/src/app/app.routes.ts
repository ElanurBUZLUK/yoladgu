import { Routes } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard';
import { MathQuestionComponent } from './math-question/math-question';
import { EnglishQuestionComponent } from './english-question/english-question';

export const routes: Routes = [
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'math-question', component: MathQuestionComponent },
  { path: 'english-question', component: EnglishQuestionComponent },
  { path: '**', redirectTo: '/dashboard' }
];
