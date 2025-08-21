import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EnglishQuestion } from './english-question';

describe('EnglishQuestion', () => {
  let component: EnglishQuestion;
  let fixture: ComponentFixture<EnglishQuestion>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EnglishQuestion]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EnglishQuestion);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
