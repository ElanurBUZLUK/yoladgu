import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.scss']
})
export class RegisterComponent {
  formData = {
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    full_name: '',
    grade: null as number | null
  };
  
  isLoading = false;
  errorMessage = '';
  successMessage = '';

  grades = [9, 10, 11, 12];

  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  onSubmit() {
    if (!this.validateForm()) {
      return;
    }

    this.isLoading = true;
    this.errorMessage = '';
    this.successMessage = '';

    const { confirmPassword, ...registerData } = this.formData;

    this.authService.register(registerData).subscribe({
      next: (response) => {
        console.log('Registration successful:', response);
        this.successMessage = 'Kayıt başarılı! Giriş sayfasına yönlendiriliyorsunuz...';
        setTimeout(() => {
          this.router.navigate(['/login']);
        }, 2000);
      },
      error: (error) => {
        console.error('Registration error:', error);
        this.errorMessage = error.error?.detail || 'Kayıt olurken bir hata oluştu.';
        this.isLoading = false;
      }
    });
  }

  validateForm(): boolean {
    if (!this.formData.username || !this.formData.email || !this.formData.password) {
      this.errorMessage = 'Lütfen tüm zorunlu alanları doldurun.';
      return false;
    }

    if (this.formData.password !== this.formData.confirmPassword) {
      this.errorMessage = 'Şifreler eşleşmiyor.';
      return false;
    }

    if (this.formData.password.length < 6) {
      this.errorMessage = 'Şifre en az 6 karakter olmalıdır.';
      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(this.formData.email)) {
      this.errorMessage = 'Geçerli bir e-posta adresi girin.';
      return false;
    }

    return true;
  }

  goToLogin() {
    this.router.navigate(['/login']);
  }
}