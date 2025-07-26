import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { StudentService } from '../features/student/services/student.service';

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
  loading = true;

  constructor(private router: Router, private studentService: StudentService) {}

  ngOnInit(): void {
    this.fetchAll();
  }

  fetchAll() {
    this.loading = true;
    this.studentService.getProfile().subscribe(profile => {
      this.profile = profile;
      this.studentService.getLevel().subscribe(level => {
        this.level = level;
        this.studentService.getSubjects().subscribe(subjects => {
          this.subjects = subjects;
          this.studentService.getStudyPlans(profile.id).subscribe(plans => {
            this.studyPlans = plans;
            this.loading = false;
          });
        });
      });
    });
  }

  goToSolveQuestion() {
    this.router.navigate(['/solve-question']);
  }
} 