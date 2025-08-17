import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule } from '@angular/router';
import { Subscription } from 'rxjs';
import { AuthService } from '../../../core/services/auth.service';

interface User {
  id: number;
  username: string;
  email: string;
  full_name?: string;
  grade?: string;
  is_active: boolean;
}

@Component({
  selector: 'app-navigation',
  standalone: true,
  imports: [CommonModule, RouterModule],
  template: `
    <nav class="navbar">
      <div class="nav-container">
        <!-- Logo/Brand -->
        <div class="nav-brand">
          <a routerLink="/" class="brand-link">
            <span class="brand-text">Yoladgu</span>
          </a>
        </div>

        <!-- Navigation Menu -->
        <div class="nav-menu" [class.active]="isMenuOpen">
          <a routerLink="/" 
             routerLinkActive="active" 
             [routerLinkActiveOptions]="{exact: true}"
             class="nav-link"
             (click)="closeMenu()">
            <i class="icon-dashboard"></i>
            <span>Dashboard</span>
          </a>
          
          <a routerLink="/solve-question" 
             routerLinkActive="active"
             class="nav-link solve-btn"
             (click)="closeMenu()">
            <i class="icon-play"></i>
            <span>Soru Çöz</span>
          </a>

          <div class="nav-divider"></div>

          <!-- User Menu -->
          <div class="user-menu" *ngIf="currentUser">
            <div class="user-info">
              <div class="user-avatar">
                {{ getUserInitials() }}
              </div>
              <div class="user-details">
                <span class="user-name">{{ currentUser.full_name || currentUser.username }}</span>
                <span class="user-grade" *ngIf="currentUser.grade">{{ currentUser.grade }}. Sınıf</span>
              </div>
            </div>
            
            <button class="logout-btn" (click)="logout()">
              <i class="icon-logout"></i>
              <span>Çıkış</span>
            </button>
          </div>
        </div>

        <!-- Mobile Menu Toggle -->
        <button class="menu-toggle" 
                (click)="toggleMenu()"
                [class.active]="isMenuOpen">
          <span class="hamburger-line"></span>
          <span class="hamburger-line"></span>
          <span class="hamburger-line"></span>
        </button>
      </div>
    </nav>
  `,
  styles: [`
    .navbar {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      padding: 0;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      position: sticky;
      top: 0;
      z-index: 1000;
    }

    .nav-container {
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 1rem;
      height: 64px;
    }

    .nav-brand .brand-link {
      color: white;
      text-decoration: none;
      font-size: 1.5rem;
      font-weight: bold;
    }

    .brand-text {
      background: linear-gradient(45deg, #fff, #f0f8ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .nav-menu {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .nav-link {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      color: rgba(255,255,255,0.9);
      text-decoration: none;
      border-radius: 6px;
      transition: all 0.3s ease;
      font-weight: 500;
    }

    .nav-link:hover {
      background: rgba(255,255,255,0.1);
      color: white;
    }

    .nav-link.active {
      background: rgba(255,255,255,0.2);
      color: white;
    }

    .solve-btn {
      background: rgba(255,255,255,0.15);
      border: 1px solid rgba(255,255,255,0.3);
    }

    .solve-btn:hover {
      background: rgba(255,255,255,0.25);
      transform: translateY(-1px);
    }

    .nav-divider {
      width: 1px;
      height: 30px;
      background: rgba(255,255,255,0.3);
      margin: 0 0.5rem;
    }

    .user-menu {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .user-info {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .user-avatar {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: rgba(255,255,255,0.2);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 600;
      font-size: 0.9rem;
    }

    .user-details {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
    }

    .user-name {
      color: white;
      font-weight: 500;
      font-size: 0.9rem;
    }

    .user-grade {
      color: rgba(255,255,255,0.7);
      font-size: 0.8rem;
    }

    .logout-btn {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      background: rgba(255,255,255,0.1);
      border: 1px solid rgba(255,255,255,0.3);
      color: white;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.3s ease;
      font-size: 0.9rem;
    }

    .logout-btn:hover {
      background: rgba(255,255,255,0.2);
    }

    .menu-toggle {
      display: none;
      flex-direction: column;
      background: none;
      border: none;
      cursor: pointer;
      padding: 0.5rem;
    }

    .hamburger-line {
      width: 25px;
      height: 3px;
      background: white;
      margin: 2px 0;
      transition: 0.3s;
    }

    /* Icons */
    [class^="icon-"] {
      font-size: 1.1rem;
    }

    .icon-dashboard::before { content: "🏠"; }
    .icon-play::before { content: "▶️"; }
    .icon-logout::before { content: "🚪"; }

    /* Mobile Responsive */
    @media (max-width: 768px) {
      .menu-toggle {
        display: flex;
      }

      .nav-menu {
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        flex-direction: column;
        align-items: stretch;
        padding: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transform: translateY(-100%);
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
      }

      .nav-menu.active {
        transform: translateY(0);
        opacity: 1;
        visibility: visible;
      }

      .nav-link {
        justify-content: center;
        padding: 1rem;
        margin: 0.25rem 0;
      }

      .nav-divider {
        width: 100%;
        height: 1px;
        margin: 0.5rem 0;
      }

      .user-menu {
        flex-direction: column;
        align-items: stretch;
        gap: 1rem;
      }

      .user-info {
        justify-content: center;
        padding: 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
      }

      .logout-btn {
        justify-content: center;
      }
    }
  `]
})
export class NavigationComponent implements OnInit, OnDestroy {
  currentUser: User | null = null;
  isMenuOpen = false;
  private userSubscription?: Subscription;

  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.userSubscription = this.authService.currentUser$.subscribe(user => {
      this.currentUser = user;
    });
  }

  ngOnDestroy(): void {
    if (this.userSubscription) {
      this.userSubscription.unsubscribe();
    }
  }

  getUserInitials(): string {
    if (!this.currentUser) return 'U';
    
    const name = this.currentUser.full_name || this.currentUser.username;
    const words = name.split(' ');
    
    if (words.length >= 2) {
      return (words[0][0] + words[1][0]).toUpperCase();
    }
    
    return name.substring(0, 2).toUpperCase();
  }

  toggleMenu(): void {
    this.isMenuOpen = !this.isMenuOpen;
  }

  closeMenu(): void {
    this.isMenuOpen = false;
  }

  logout(): void {
    this.authService.logout();
    this.closeMenu();
  }
}