import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { ErrorHandlerService, ErrorMessage } from '../../../core/services/error-handler.service';

@Component({
  selector: 'app-error-toast',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './error-toast.component.html',
  styleUrls: ['./error-toast.component.scss']
})
export class ErrorToastComponent implements OnInit, OnDestroy {
  errors: ErrorMessage[] = [];
  private subscription?: Subscription;

  constructor(private errorHandler: ErrorHandlerService) {}

  ngOnInit(): void {
    this.subscription = this.errorHandler.errors$.subscribe(errors => {
      this.errors = errors;
    });
  }

  ngOnDestroy(): void {
    if (this.subscription) {
      this.subscription.unsubscribe();
    }
  }

  removeError(id: string): void {
    this.errorHandler.removeError(id);
  }

  getIcon(type: string): string {
    switch (type) {
      case 'error': return '❌';
      case 'warning': return '⚠️';
      case 'info': return 'ℹ️';
      case 'success': return '✅';
      default: return 'ℹ️';
    }
  }

  getTypeClass(type: string): string {
    return `toast-${type}`;
  }
}