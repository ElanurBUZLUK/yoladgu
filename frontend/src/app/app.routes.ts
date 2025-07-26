import { Routes } from '@angular/router';
import { DashboardComponent } from './pages/dashboard';
import { SolveQuestionComponent } from './solve-question/solve-question';
import { authGuard } from './core/guards/auth-guard';

export const routes: Routes = [
  { path: '', component: DashboardComponent },
  { path: 'solve', component: SolveQuestionComponent, canActivate: [authGuard] },
  { path: 'solve-question', component: SolveQuestionComponent, canActivate: [authGuard] },
];
