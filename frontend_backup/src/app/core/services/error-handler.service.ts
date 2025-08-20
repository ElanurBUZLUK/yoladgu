import { Injectable } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { BehaviorSubject } from 'rxjs';

export interface ErrorMessage {
  id: string;
  message: string;
  type: 'error' | 'warning' | 'info' | 'success';
  duration?: number;
  timestamp: Date;
}

@Injectable({
  providedIn: 'root'
})
export class ErrorHandlerService {
  private errorsSubject = new BehaviorSubject<ErrorMessage[]>([]);
  public errors$ = this.errorsSubject.asObservable();

  private generateId(): string {
    return Math.random().toString(36).substr(2, 9);
  }

  handleHttpError(error: HttpErrorResponse): string {
    let errorMessage = 'Bilinmeyen bir hata oluştu.';
    
    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `İstemci hatası: ${error.error.message}`;
    } else {
      // Server-side error
      switch (error.status) {
        case 400:
          errorMessage = error.error?.detail || 'Geçersiz istek.';
          break;
        case 401:
          errorMessage = 'Oturum süreniz dolmuş. Lütfen tekrar giriş yapın.';
          break;
        case 403:
          errorMessage = 'Bu işlem için yetkiniz bulunmuyor.';
          break;
        case 404:
          errorMessage = 'İstenen kaynak bulunamadı.';
          break;
        case 422:
          errorMessage = this.handleValidationErrors(error.error);
          break;
        case 500:
          errorMessage = 'Sunucu hatası. Lütfen daha sonra tekrar deneyin.';
          break;
        case 503:
          errorMessage = 'Servis şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.';
          break;
        default:
          errorMessage = `Sunucu hatası (${error.status}): ${error.error?.detail || error.message}`;
      }
    }

    this.showError(errorMessage);
    return errorMessage;
  }

  private handleValidationErrors(errorData: any): string {
    if (errorData?.detail && Array.isArray(errorData.detail)) {
      const messages = errorData.detail.map((item: any) => 
        `${item.loc?.join(' -> ') || 'Alan'}: ${item.msg}`
      );
      return messages.join(', ');
    }
    return errorData?.detail || 'Doğrulama hatası.';
  }

  showError(message: string, duration: number = 5000): void {
    this.addMessage(message, 'error', duration);
  }

  showWarning(message: string, duration: number = 4000): void {
    this.addMessage(message, 'warning', duration);
  }

  showInfo(message: string, duration: number = 3000): void {
    this.addMessage(message, 'info', duration);
  }

  showSuccess(message: string, duration: number = 3000): void {
    this.addMessage(message, 'success', duration);
  }

  private addMessage(message: string, type: ErrorMessage['type'], duration?: number): void {
    const error: ErrorMessage = {
      id: this.generateId(),
      message,
      type,
      duration,
      timestamp: new Date()
    };

    const currentErrors = this.errorsSubject.value;
    this.errorsSubject.next([...currentErrors, error]);

    if (duration && duration > 0) {
      setTimeout(() => {
        this.removeError(error.id);
      }, duration);
    }
  }

  removeError(id: string): void {
    const currentErrors = this.errorsSubject.value;
    const filteredErrors = currentErrors.filter(error => error.id !== id);
    this.errorsSubject.next(filteredErrors);
  }

  clearAllErrors(): void {
    this.errorsSubject.next([]);
  }
}