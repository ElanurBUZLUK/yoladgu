import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent {
  constructor(private router: Router) {}

  goToMath() {
    this.router.navigate(['/math']);
  }

  goToEnglish() {
    this.router.navigate(['/english']);
  }
}
