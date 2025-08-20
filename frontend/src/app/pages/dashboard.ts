import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { StudentService } from '../features/student/services/student.service';
import { AuthService } from '../core/services/auth.service';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.html',
  styleUrls: ['./dashboard.scss']
})
export class DashboardComponent implements OnInit {
  profile: any;
  level: any;
  subjects: any[] = [];
  studyPlans: any[] = [];
  analytics: any = {};
  quizHistory: any[] = [];
  performanceStats: any = {};
  loading = true;

  constructor(
    private router: Router, 
    private studentService: StudentService,
    private authService: AuthService
  ) {}

  ngOnInit(): void {
    this.fetchAll();
  }

  fetchAll() {
    this.loading = true;
    let completedRequests = 0;
    const totalRequests = 6;
    
    const checkComplete = () => {
      completedRequests++;
      if (completedRequests >= totalRequests) {
        this.loading = false;
      }
    };

    // Temel profil verileri
    this.studentService.getProfile().subscribe({
      next: (profile) => {
        this.profile = profile;
        checkComplete();
        
        // Study plans için user ID gerekiyor
        this.studentService.getStudyPlans(profile.id).subscribe({
          next: (plans) => {
            this.studyPlans = plans;
            checkComplete();
          },
          error: () => {
            this.studyPlans = this.getMockStudyPlans();
            checkComplete();
          }
        });
      },
      error: () => {
        this.profile = this.getMockProfile();
        this.studyPlans = this.getMockStudyPlans();
        checkComplete();
        checkComplete(); // İki request için
      }
    });

    // Level bilgisi
    this.studentService.getLevel().subscribe({
      next: (level) => {
        this.level = level;
        checkComplete();
      },
      error: () => {
        this.level = this.getMockLevel();
        checkComplete();
      }
    });

    // Subjects
    this.studentService.getSubjects().subscribe({
      next: (subjects) => {
        this.subjects = subjects;
        checkComplete();
      },
      error: () => {
        this.subjects = this.getMockSubjects();
        checkComplete();
      }
    });

    // Analytics
    this.studentService.getStudentAnalytics().subscribe({
      next: (analytics) => {
        this.analytics = analytics;
        checkComplete();
      },
      error: () => {
        this.analytics = this.getMockAnalytics();
        checkComplete();
      }
    });

    // Quiz History
    this.studentService.getQuizHistory(5).subscribe({
      next: (history) => {
        this.quizHistory = history;
        checkComplete();
      },
      error: () => {
        this.quizHistory = this.getMockQuizHistory();
        checkComplete();
      }
    });
  }

  // Mock data methods
  private getMockProfile() {
    return {
      id: 1,
      username: 'demo_user',
      full_name: 'Demo Kullanıcı',
      grade: 11,
      solved_questions: 156,
      accuracy: 78,
      study_time: '24 saat',
      last_login: '2 saat önce'
    };
  }

  private getMockLevel() {
    return {
      level: 8,
      max_level: 20,
      experience: 1240,
      next_level_exp: 1500
    };
  }

  private getMockSubjects() {
    return [
      { name: 'Matematik', topic_count: 25 },
      { name: 'Fizik', topic_count: 18 },
      { name: 'Kimya', topic_count: 15 },
      { name: 'Biyoloji', topic_count: 12 }
    ];
  }

  private getMockStudyPlans() {
    return [
      {
        title: 'Trigonometri Temelleri',
        description: 'Sinüs, kosinüs ve tanjant fonksiyonlarını öğren'
      },
      {
        title: 'İkinci Derece Denklemler',
        description: 'Diskriminant ve kök formülü ile problem çözme'
      },
      {
        title: 'Geometri Problemleri',
        description: 'Alan ve çevre hesaplamaları'
      }
    ];
  }

  private getMockAnalytics() {
    return {
      total_sessions: 24,
      average_accuracy: 76.5,
      total_study_time: 1440, // minutes
      improvement_rate: 12.3,
      streak_days: 7,
      favorite_subject: 'Matematik'
    };
  }

  private getMockQuizHistory() {
    return [
      {
        date: '2024-01-15',
        questions: 10,
        correct: 8,
        accuracy: 80,
        time: 780 // seconds
      },
      {
        date: '2024-01-14',
        questions: 15,
        correct: 11,
        accuracy: 73,
        time: 1200
      }
    ];
  }

  goToSolveQuestion() {
    this.router.navigate(['/solve-question']);
  }

  logout() {
    this.authService.logout();
  }

  openSubject(subject: any) {
    console.log('Opening subject:', subject);
    
    // Navigate to solve questions with subject filter
    this.router.navigate(['/solve-question'], { 
      queryParams: { 
        subject_id: subject.id,
        subject_name: subject.name 
      } 
    });
  }

  startPlan(planTitle: string) {
    console.log('Starting plan:', planTitle);
    
    // Find the plan details
    const plan = this.studyPlans.find(p => p.title === planTitle);
    if (!plan) {
      console.error('Plan not found:', planTitle);
      return;
    }
    
    // Navigate to solve questions with plan context
    this.router.navigate(['/solve-question'], { 
      queryParams: { 
        plan_title: planTitle,
        plan_description: plan.description,
        mode: 'study_plan'
      } 
    });
  }
} 