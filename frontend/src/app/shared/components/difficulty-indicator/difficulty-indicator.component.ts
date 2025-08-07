import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-difficulty-indicator',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="difficulty-container" [class]="difficultyClass">
      <div class="difficulty-badge">
        <span class="difficulty-text">{{ difficultyText }}</span>
        <div class="difficulty-stars">
          <span 
            *ngFor="let star of stars; let i = index" 
            class="star"
            [class.filled]="i < difficultyLevel"
            [class.half]="i === difficultyLevel - 0.5">
            ★
          </span>
        </div>
      </div>
      
      <div class="difficulty-details" *ngIf="showDetails">
        <div class="detail-item">
          <span class="detail-label">Zorluk:</span>
          <span class="detail-value">{{ difficultyLevel }}/5</span>
        </div>
        <div class="detail-item" *ngIf="successRate !== undefined">
          <span class="detail-label">Başarı Oranı:</span>
          <span class="detail-value">{{ successRate }}%</span>
        </div>
        <div class="detail-item" *ngIf="averageTime !== undefined">
          <span class="detail-label">Ortalama Süre:</span>
          <span class="detail-value">{{ formatTime(averageTime) }}</span>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .difficulty-container {
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      padding: 0.5rem;
      border-radius: 8px;
      background: #f8f9fa;
      border: 2px solid transparent;
      transition: all 0.3s ease;
    }

    .difficulty-container.easy {
      border-color: #28a745;
      background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }

    .difficulty-container.medium {
      border-color: #ffc107;
      background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }

    .difficulty-container.hard {
      border-color: #fd7e14;
      background: linear-gradient(135deg, #ffe8d6 0%, #ffd8a8 100%);
    }

    .difficulty-container.expert {
      border-color: #dc3545;
      background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    }

    .difficulty-badge {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.5rem;
    }

    .difficulty-text {
      font-weight: 600;
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .difficulty-container.easy .difficulty-text {
      color: #155724;
    }

    .difficulty-container.medium .difficulty-text {
      color: #856404;
    }

    .difficulty-container.hard .difficulty-text {
      color: #a04000;
    }

    .difficulty-container.expert .difficulty-text {
      color: #721c24;
    }

    .difficulty-stars {
      display: flex;
      gap: 2px;
    }

    .star {
      font-size: 1.2rem;
      color: #dee2e6;
      transition: color 0.3s ease;
    }

    .star.filled {
      color: #ffc107;
    }

    .star.half {
      color: #ffc107;
      position: relative;
    }

    .star.half::after {
      content: '★';
      position: absolute;
      left: 0;
      color: #dee2e6;
      clip-path: polygon(0 0, 50% 0, 50% 100%, 0 100%);
    }

    .difficulty-details {
      margin-top: 0.5rem;
      padding: 0.5rem;
      background: rgba(255, 255, 255, 0.7);
      border-radius: 4px;
      font-size: 0.8rem;
    }

    .detail-item {
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.2rem;
    }

    .detail-item:last-child {
      margin-bottom: 0;
    }

    .detail-label {
      color: #666;
      font-weight: 500;
    }

    .detail-value {
      font-weight: 600;
      color: #333;
    }

    /* Hover effects */
    .difficulty-container:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    /* Responsive */
    @media (max-width: 768px) {
      .difficulty-container {
        padding: 0.3rem;
      }
      
      .difficulty-text {
        font-size: 0.8rem;
      }
      
      .star {
        font-size: 1rem;
      }
    }

    /* Animation for expert difficulty */
    .difficulty-container.expert .star.filled {
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.7; }
    }
  `]
})
export class DifficultyIndicatorComponent {
  @Input() difficultyLevel: number = 1; // 1-5 arası
  @Input() showDetails: boolean = false;
  @Input() successRate?: number;
  @Input() averageTime?: number; // saniye cinsinden

  get difficultyClass(): string {
    if (this.difficultyLevel <= 1) return 'easy';
    if (this.difficultyLevel <= 2) return 'medium';
    if (this.difficultyLevel <= 3) return 'hard';
    return 'expert';
  }

  get difficultyText(): string {
    if (this.difficultyLevel <= 1) return 'Kolay';
    if (this.difficultyLevel <= 2) return 'Orta';
    if (this.difficultyLevel <= 3) return 'Zor';
    if (this.difficultyLevel <= 4) return 'Çok Zor';
    return 'Uzman';
  }

  get stars(): number[] {
    return Array.from({ length: 5 }, (_, i) => i + 1);
  }

  formatTime(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    
    if (minutes > 0) {
      return `${minutes}:${secs.toString().padStart(2, '0')}`;
    } else {
      return `${secs}s`;
    }
  }
} 