export interface EnglishQuestion {
  id: string;
  content: string;
  options: string[];
  correct_answer: string;
  difficulty_level: number;
  topic_category?: string | null;
}

export interface EnglishQuestionResponse {
  success: boolean;
  question?: EnglishQuestion | null;
  generation_info?: Record<string, unknown>;
}

export interface EnglishQuestionRequest {
  student_id: string;
  k: number; // 1..10
}
