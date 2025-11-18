import { Routes } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard';
import { MathQuestionComponent } from './math-question/math-question';
import { EnglishQuestionComponent } from './english-question/english-question';
import { LoginComponent } from './login/login';
import { authGuard } from './auth.guard';

export const routes: Routes = [
  { path: '', redirectTo: '/login', pathMatch: 'full' },
  { path: 'login', component: LoginComponent },
  { path: 'dashboard', component: DashboardComponent, canActivate: [authGuard] },
  { path: 'math-question', component: MathQuestionComponent, canActivate: [authGuard] },
  { path: 'english-question', component: EnglishQuestionComponent, canActivate: [authGuard] },
  { path: '**', redirectTo: '/login' }
];
