export interface MathQuestion {
  id: string;
  content: string;
  options: string[];
  correct_answer: string;
  difficulty_level: number;
  topic_category?: string | null;
}

export interface MathQuestionResponse {
  success: boolean;
  question?: MathQuestion | null;
  generation_info?: Record<string, unknown>;
}

export interface MathQuestionRequest {
  user_id: string;
  k: number; // 1..10
}
