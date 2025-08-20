import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SolveQuestion } from './solve-question';

describe('SolveQuestion', () => {
  let component: SolveQuestion;
  let fixture: ComponentFixture<SolveQuestion>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SolveQuestion]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SolveQuestion);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
