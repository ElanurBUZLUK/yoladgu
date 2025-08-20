import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { ErrorToastComponent } from './shared/components/error-toast/error-toast.component';
import { ApiDebugComponent } from './shared/components/api-debug/api-debug.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, ErrorToastComponent, ApiDebugComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  protected readonly title = signal('frontend');
}
