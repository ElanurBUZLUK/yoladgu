import { Routes } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard.component';
import { MathQuestionComponent } from './math/math-question.component';
import { EnglishQuestionComponent } from './english/english-question.component';
import { LoginComponent } from './pages/login/login.component';
import { RegisterComponent } from './pages/register/register.component';

export const routes: Routes = [
  { path: '', component: DashboardComponent },
  { path: 'login', component: LoginComponent },
  { path: 'register', component: RegisterComponent },
  { path: 'math', component: MathQuestionComponent },
  { path: 'english', component: EnglishQuestionComponent },
  { path: '**', redirectTo: '' }
];
