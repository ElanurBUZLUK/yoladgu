import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiConfig } from '../../../core/config/api.config';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-api-debug',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="api-debug" *ngIf="showDebug">
      <h4>🔧 API Configuration Debug</h4>
      <div class="debug-info">
        <p><strong>Environment:</strong> {{ config.production ? 'Production' : 'Development' }}</p>
        <p><strong>API URL:</strong> {{ config.apiUrl }}</p>
        <p><strong>Backend URL:</strong> {{ config.backendUrl }}</p>
        <p><strong>Current Base URL:</strong> {{ config.currentBaseUrl }}</p>
      </div>
      
      <div class="test-section">
        <h5>🧪 API Test Endpoints:</h5>
        <ul>
          <li>Users: {{ getTestUrl('users/me') }}</li>
          <li>Questions: {{ getTestUrl('recommendations/next-question') }}</li>
          <li>Auth: {{ getTestUrl('auth/login') }}</li>
        </ul>
      </div>
      
      <button (click)="testConnection()" class="test-btn">
        {{ testing ? 'Testing...' : 'Test API Connection' }}
      </button>
      
      <div *ngIf="testResult" class="test-result" [class.success]="testResult.success" [class.error]="!testResult.success">
        {{ testResult.message }}
      </div>
      
      <button (click)="toggleDebug()" class="close-btn">×</button>
    </div>
    
    <button *ngIf="!showDebug" (click)="toggleDebug()" class="debug-toggle">
      🔧 Debug API
    </button>
  `,
  styles: [`
    .api-debug {
      position: fixed;
      top: 20px;
      right: 20px;
      background: rgba(15, 23, 42, 0.95);
      border: 2px solid #3b82f6;
      border-radius: 10px;
      padding: 20px;
      max-width: 400px;
      z-index: 1000;
      color: white;
      font-size: 14px;
    }
    
    .debug-info p {
      margin: 5px 0;
      font-family: monospace;
    }
    
    .test-section {
      margin: 15px 0;
    }
    
    .test-section ul {
      margin: 10px 0;
      padding-left: 20px;
    }
    
    .test-section li {
      margin: 5px 0;
      font-family: monospace;
      font-size: 12px;
    }
    
    .test-btn, .close-btn, .debug-toggle {
      background: #3b82f6;
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 5px;
      cursor: pointer;
      margin: 5px;
    }
    
    .close-btn {
      position: absolute;
      top: 10px;
      right: 10px;
      width: 30px;
      height: 30px;
      border-radius: 50%;
      padding: 0;
    }
    
    .debug-toggle {
      position: fixed;
      bottom: 20px;
      right: 20px;
      z-index: 999;
    }
    
    .test-result {
      margin: 10px 0;
      padding: 10px;
      border-radius: 5px;
    }
    
    .test-result.success {
      background: rgba(16, 185, 129, 0.2);
      border: 1px solid #10b981;
      color: #10b981;
    }
    
    .test-result.error {
      background: rgba(239, 68, 68, 0.2);
      border: 1px solid #ef4444;
      color: #ef4444;
    }
  `]
})
export class ApiDebugComponent {
  showDebug = false;
  config = ApiConfig.getConfig();
  testing = false;
  testResult: { success: boolean; message: string } | null = null;

  constructor(private http: HttpClient) {}

  toggleDebug() {
    this.showDebug = !this.showDebug;
  }

  getTestUrl(endpoint: string): string {
    return ApiConfig.getApiUrl(endpoint);
  }

  async testConnection() {
    this.testing = true;
    this.testResult = null;

    try {
      // Try a simple endpoint that doesn't require authentication
      const testUrl = ApiConfig.getApiUrl('health'); // Assuming there's a health endpoint
      
      this.http.get(testUrl).subscribe({
        next: () => {
          this.testResult = {
            success: true,
            message: '✅ API connection successful!'
          };
          this.testing = false;
        },
        error: (error) => {
          // Check if it's a 404 (endpoint exists but not found) or connection error
          if (error.status === 404) {
            this.testResult = {
              success: true,
              message: '✅ Backend reachable (404 expected for /health)'
            };
          } else if (error.status === 0) {
            this.testResult = {
              success: false,
              message: '❌ Cannot reach backend. Check if backend is running on localhost:8000'
            };
          } else {
            this.testResult = {
              success: false,
              message: `❌ API error: ${error.status} ${error.statusText}`
            };
          }
          this.testing = false;
        }
      });
    } catch (error) {
      this.testResult = {
        success: false,
        message: '❌ Connection test failed'
      };
      this.testing = false;
    }
  }
}