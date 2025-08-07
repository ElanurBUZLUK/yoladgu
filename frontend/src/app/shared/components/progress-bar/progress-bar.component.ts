import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-progress-bar',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="progress-container">
      <div class="progress-header">
        <span class="progress-label">{{ label }}</span>
        <span class="progress-value">{{ current }}/{{ total }}</span>
        <span class="progress-percentage">{{ percentage }}%</span>
      </div>
      
      <div class="progress-bar-wrapper">
        <div 
          class="progress-bar" 
          [style.width.%]="percentage"
          [class.animated]="animated"
          [class.success]="isSuccess"
          [class.warning]="isWarning"
          [class.danger]="isDanger">
        </div>
      </div>
      
      <div class="progress-details" *ngIf="showDetails">
        <div class="detail-item" *ngIf="correctAnswers !== undefined">
          <span class="detail-label">Doğru:</span>
          <span class="detail-value correct">{{ correctAnswers }}</span>
        </div>
        <div class="detail-item" *ngIf="wrongAnswers !== undefined">
          <span class="detail-label">Yanlış:</span>
          <span class="detail-value wrong">{{ wrongAnswers }}</span>
        </div>
        <div class="detail-item" *ngIf="accuracy !== undefined">
          <span class="detail-label">Başarı:</span>
          <span class="detail-value accuracy">{{ accuracy }}%</span>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .progress-container {
      width: 100%;
      margin: 1rem 0;
    }

    .progress-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
      font-size: 0.9rem;
      color: #666;
    }

    .progress-label {
      font-weight: 500;
      color: #333;
    }

    .progress-value {
      font-weight: 600;
      color: #007bff;
    }

    .progress-percentage {
      font-weight: 600;
      color: #28a745;
    }

    .progress-bar-wrapper {
      width: 100%;
      height: 12px;
      background-color: #e9ecef;
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }

    .progress-bar {
      height: 100%;
      background: linear-gradient(90deg, #007bff 0%, #0056b3 100%);
      border-radius: 6px;
      transition: width 0.3s ease;
      position: relative;
    }

    .progress-bar.animated {
      transition: width 0.5s ease-in-out;
    }

    .progress-bar.success {
      background: linear-gradient(90deg, #28a745 0%, #1e7e34 100%);
    }

    .progress-bar.warning {
      background: linear-gradient(90deg, #ffc107 0%, #e0a800 100%);
    }

    .progress-bar.danger {
      background: linear-gradient(90deg, #dc3545 0%, #c82333 100%);
    }

    .progress-bar::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(255, 255, 255, 0.3) 50%,
        transparent 100%
      );
      animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }

    .progress-details {
      display: flex;
      justify-content: space-around;
      margin-top: 0.5rem;
      padding: 0.5rem;
      background-color: #f8f9fa;
      border-radius: 4px;
    }

    .detail-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
    }

    .detail-label {
      font-size: 0.8rem;
      color: #666;
      margin-bottom: 0.2rem;
    }

    .detail-value {
      font-weight: 600;
      font-size: 1.1rem;
    }

    .detail-value.correct {
      color: #28a745;
    }

    .detail-value.wrong {
      color: #dc3545;
    }

    .detail-value.accuracy {
      color: #007bff;
    }

    /* Responsive */
    @media (max-width: 768px) {
      .progress-header {
        flex-direction: column;
        gap: 0.5rem;
        text-align: center;
      }
      
      .progress-details {
        flex-direction: column;
        gap: 0.5rem;
      }
    }
  `]
})
export class ProgressBarComponent {
  @Input() current: number = 0;
  @Input() total: number = 100;
  @Input() label: string = 'İlerleme';
  @Input() animated: boolean = true;
  @Input() showDetails: boolean = false;
  @Input() correctAnswers?: number;
  @Input() wrongAnswers?: number;
  @Input() accuracy?: number;
  @Input() successThreshold: number = 80;
  @Input() warningThreshold: number = 60;
  
  @Output() progressComplete = new EventEmitter<void>();
  @Output() progressUpdate = new EventEmitter<number>();

  get percentage(): number {
    if (this.total === 0) return 0;
    return Math.round((this.current / this.total) * 100);
  }

  get isSuccess(): boolean {
    return this.percentage >= this.successThreshold;
  }

  get isWarning(): boolean {
    return this.percentage >= this.warningThreshold && this.percentage < this.successThreshold;
  }

  get isDanger(): boolean {
    return this.percentage < this.warningThreshold;
  }

  ngOnInit() {
    if (this.percentage === 100) {
      this.progressComplete.emit();
    }
    this.progressUpdate.emit(this.percentage);
  }

  ngOnChanges() {
    this.progressUpdate.emit(this.percentage);
  }
} 