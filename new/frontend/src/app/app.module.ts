import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { AppRoutingModule } from './app-routing.module';

// Import Angular Material modules (assuming they are installed)
import { MatButtonModule } from '@angular/material/button';

import { AppComponent } from './app.component'; // Assuming a root AppComponent
import { DashboardComponent } from './dashboard/dashboard.component';
import { MathQuestionComponent } from './math/math-question.component';
import { EnglishQuestionComponent } from './english/english-question.component';

@NgModule({
  declarations: [
    AppComponent,
    DashboardComponent,
    MathQuestionComponent,
    EnglishQuestionComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    MatButtonModule // Add MatButtonModule for Angular Material buttons
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
