import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MathQuestion } from './math-question';

describe('MathQuestion', () => {
  let component: MathQuestion;
  let fixture: ComponentFixture<MathQuestion>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MathQuestion]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MathQuestion);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
