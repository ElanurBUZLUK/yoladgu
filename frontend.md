Harika 👍 O zaman sadece dashboard değil, **tam akışı** kuralım:

### 📂 Önerilen Angular Yapısı

```
src/app/
 ├── app-routing.module.ts
 ├── app.module.ts
 ├── dashboard/
 │    ├── dashboard.component.ts
 │    ├── dashboard.component.html
 │    └── dashboard.component.css
 ├── math/
 │    ├── math-question.component.ts
 │    ├── math-question.component.html
 │    └── math-question.component.css
 ├── english/
 │    ├── english-question.component.ts
 │    ├── english-question.component.html
 │    └── english-question.component.css
 └── services/
      └── api.service.ts
```

---

### 🟦 DashboardComponent

👉 İki buton var, biri **Matematik** biri **İngilizce** ekranına yönlendiriyor.

**dashboard.component.html**

```html
<div class="dashboard">
  <h2>Öğrenci Dashboard</h2>

  <div class="button-row">
    <button mat-raised-button color="primary" (click)="goToMath()">Matematik Soru Çöz</button>
    <button mat-raised-button color="accent" (click)="goToEnglish()">İngilizce Soru Çöz</button>
  </div>
</div>
```

**dashboard.component.ts**

```ts
import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-dashboard',
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
```

---

### 🟦 MathQuestionComponent

👉 Matematik sorularını `/api/v1/math/recommend` endpoint’inden çeker.

**math-question.component.html**

```html
<div class="math-container">
  <h2>Matematik Soru Çöz</h2>

  <div *ngIf="question">
    <p>{{ question.text }}</p>

    <div *ngFor="let option of question.options">
      <button mat-stroked-button (click)="submitAnswer(option)">
        {{ option }}
      </button>
    </div>
  </div>

  <div *ngIf="!question">
    <p>Yükleniyor...</p>
  </div>
</div>
```

**math-question.component.ts**

```ts
import { Component, OnInit } from '@angular/core';
import { ApiService } from '../services/api.service';

@Component({
  selector: 'app-math-question',
  templateUrl: './math-question.component.html',
  styleUrls: ['./math-question.component.css']
})
export class MathQuestionComponent implements OnInit {
  question: any;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getMathQuestion().subscribe(res => {
      this.question = res;
    });
  }

  submitAnswer(answer: string) {
    console.log("Seçilen cevap:", answer);
    // burada backend’e answer submit eklenebilir
  }
}
```

---

### 🟦 EnglishQuestionComponent

👉 İngilizce hatalardan türetilmiş cloze sorularını `/api/v1/english/questions/generate`’ten alır.

**english-question.component.html**

```html
<div class="english-container">
  <h2>İngilizce Soru Çöz</h2>

  <div *ngIf="question">
    <p [innerHTML]="question.text"></p>

    <div *ngFor="let option of question.options">
      <button mat-stroked-button (click)="submitAnswer(option)">
        {{ option }}
      </button>
    </div>
  </div>

  <div *ngIf="!question">
    <p>Yükleniyor...</p>
  </div>
</div>
```

**english-question.component.ts**

```ts
import { Component, OnInit } from '@angular/core';
import { ApiService } from '../services/api.service';

@Component({
  selector: 'app-english-question',
  templateUrl: './english-question.component.html',
  styleUrls: ['./english-question.component.css']
})
export class EnglishQuestionComponent implements OnInit {
  question: any;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getEnglishQuestion().subscribe(res => {
      this.question = res;
    });
  }

  submitAnswer(answer: string) {
    console.log("Seçilen cevap:", answer);
    // backend’e cevabı göndermek için eklenebilir
  }
}
```

---

### 🟦 ApiService

**services/api.service.ts**

```ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private baseUrl = 'http://localhost:8000/api/v1';

  constructor(private http: HttpClient) {}

  getMathQuestion(): Observable<any> {
    return this.http.post(`${this.baseUrl}/math/recommend`, {});
  }

  getEnglishQuestion(): Observable<any> {
    return this.http.post(`${this.baseUrl}/english/questions/generate`, {});
  }
}
```

---

### 🟦 Routing

**app-routing.module.ts**

```ts
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard.component';
import { MathQuestionComponent } from './math/math-question.component';
import { EnglishQuestionComponent } from './english/english-question.component';

const routes: Routes = [
  { path: '', component: DashboardComponent },
  { path: 'math', component: MathQuestionComponent },
  { path: 'english', component: EnglishQuestionComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}
```

---

✅ Böylece:

* Uygulama açıldığında **Dashboard** gelir.
* Kullanıcı “Matematik” veya “İngilizce” butonuna basarak ilgili çözüm ekranına geçer.
* Her ekran backend ile haberleşerek soru çeker.

---

👉 İstersen sana komple **Angular modülleri ve component dosyaları (zip halinde)** hazırlayıp verebilirim. Bunu ister misin, yoksa sadece kod parçaları yeterli mi?

