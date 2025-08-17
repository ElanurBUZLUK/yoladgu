import { Routes } from '@angular/router';
import { DashboardComponent } from './pages/dashboard';
import { SolveQuestionComponent } from './solve-question/solve-question';
import { LoginComponent } from './pages/login/login.component';
import { RegisterComponent } from './pages/register/register.component';
import { authGuard } from './core/guards/auth-guard';

export const routes: Routes = [
  { path: '', component: DashboardComponent, canActivate: [authGuard] },
  { path: 'login', component: LoginComponent },
  { path: 'register', component: RegisterComponent },
  { path: 'solve', component: SolveQuestionComponent, canActivate: [authGuard] },
  { path: 'solve-question', component: SolveQuestionComponent, canActivate: [authGuard] },
  { path: '**', redirectTo: '' }
];
