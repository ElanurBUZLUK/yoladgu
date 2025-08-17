import { Component, Input, Output, EventEmitter, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { interval, Subscription } from 'rxjs';

@Component({
  selector: 'app-timer',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="timer-container" [class.pulsing]="isPulsing">
      <div class="timer-display">
        <span class="time">{{ formatTime(elapsedTime) }}</span>
        <span class="label">{{ label }}</span>
      </div>
      <div class="timer-controls" *ngIf="showControls">
        <button 
          *ngIf="!isRunning" 
          (click)="startTimer()" 
          class="btn btn-start">
          Başlat
        </button>
        <button 
          *ngIf="isRunning" 
          (click)="pauseTimer()" 
          class="btn btn-pause">
          Duraklat
        </button>
        <button 
          (click)="resetTimer()" 
          class="btn btn-reset">
          Sıfırla
        </button>
      </div>
    </div>
  `,
  styles: [`
    .timer-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 1rem;
      border-radius: 8px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
      transition: all 0.3s ease;
    }

    .timer-container.pulsing {
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.05); }
      100% { transform: scale(1); }
    }

    .timer-display {
      text-align: center;
      margin-bottom: 1rem;
    }

    .time {
      font-size: 2.5rem;
      font-weight: bold;
      font-family: 'Courier New', monospace;
      display: block;
      margin-bottom: 0.5rem;
    }

    .label {
      font-size: 0.9rem;
      opacity: 0.9;
    }

    .timer-controls {
      display: flex;
      gap: 0.5rem;
    }

    .btn {
      padding: 0.5rem 1rem;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-weight: 500;
      transition: all 0.2s ease;
    }

    .btn-start {
      background-color: #28a745;
      color: white;
    }

    .btn-start:hover {
      background-color: #218838;
    }

    .btn-pause {
      background-color: #ffc107;
      color: #212529;
    }

    .btn-pause:hover {
      background-color: #e0a800;
    }

    .btn-reset {
      background-color: #dc3545;
      color: white;
    }

    .btn-reset:hover {
      background-color: #c82333;
    }

    .warning {
      background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }

    .danger {
      background: linear-gradient(135deg, #ff4757 0%, #c44569 100%);
    }
  `]
})
export class TimerComponent implements OnInit, OnDestroy {
  @Input() duration: number = 0; // 0 = unlimited
  @Input() label: string = 'Süre';
  @Input() showControls: boolean = false;
  @Input() warningThreshold: number = 300; // 5 minutes
  @Input() dangerThreshold: number = 600; // 10 minutes
  
  @Output() timeUp = new EventEmitter<void>();
  @Output() timeUpdate = new EventEmitter<number>();
  @Output() timerStart = new EventEmitter<void>();
  @Output() timerPause = new EventEmitter<void>();
  @Output() timerReset = new EventEmitter<void>();

  elapsedTime = 0;
  isRunning = false;
  isPulsing = false;
  private timerSubscription?: Subscription;

  ngOnInit() {
    // Auto-start timer if no controls
    if (!this.showControls) {
      this.startTimer();
    }
  }

  ngOnDestroy() {
    this.stopTimer();
  }

  startTimer() {
    if (!this.isRunning) {
      this.isRunning = true;
      this.timerStart.emit();
      
      this.timerSubscription = interval(1000).subscribe(() => {
        this.elapsedTime++;
        this.timeUpdate.emit(this.elapsedTime);
        
        // Check thresholds
        if (this.elapsedTime >= this.warningThreshold) {
          this.isPulsing = true;
        }
        
        // Check if time is up
        if (this.duration > 0 && this.elapsedTime >= this.duration) {
          this.timeUp.emit();
          this.stopTimer();
        }
      });
    }
  }

  pauseTimer() {
    if (this.isRunning) {
      this.stopTimer();
      this.timerPause.emit();
    }
  }

  resetTimer() {
    this.stopTimer();
    this.elapsedTime = 0;
    this.isPulsing = false;
    this.timerReset.emit();
  }

  stopTimer() {
    if (this.timerSubscription) {
      this.timerSubscription.unsubscribe();
      this.timerSubscription = undefined;
    }
    this.isRunning = false;
  }

  formatTime(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
      return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
  }

  getTimerClass(): string {
    if (this.elapsedTime >= this.dangerThreshold) {
      return 'danger';
    } else if (this.elapsedTime >= this.warningThreshold) {
      return 'warning';
    }
    return '';
  }
} 