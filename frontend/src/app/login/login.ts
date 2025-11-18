import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';

import { ApiService } from '../services/api';
import { AuthService } from '../services/auth.service';

@Component({
  selector: 'app-login',
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './login.html',
  styleUrl: './login.scss'
})
export class LoginComponent implements OnInit {
  email = '';
  password = '';
  userId = '';
  loading = false;
  error = '';

  constructor(
    private apiService: ApiService,
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    if (this.authService.isAuthenticated()) {
      this.router.navigate(['/math-question']);
    }
  }

  submit(): void {
    if (!this.email || !this.password || !this.userId) {
      this.error = 'Email, şifre ve user_id alanları zorunludur.';
      return;
    }

    this.loading = true;
    this.error = '';

    this.apiService.login(this.email, this.password).subscribe({
      next: (response) => {
        const accessToken = response?.access_token;
        if (!accessToken) {
          this.error = 'Access token alınamadı.';
          this.loading = false;
          return;
        }

        const refreshToken = response?.refresh_token || null;
        this.authService.setSession(accessToken, refreshToken, this.userId);
        this.loading = false;
        this.router.navigate(['/math-question']);
      },
      error: (err) => {
        this.error = err?.error?.detail || 'Giriş başarısız. Lütfen bilgileri kontrol edin.';
        this.loading = false;
      }
    });
  }
}
