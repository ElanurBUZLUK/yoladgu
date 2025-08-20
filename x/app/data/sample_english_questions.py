"""
Sample English questions for seeding the database
"""

from app.models.question import Subject, QuestionType, SourceType

SAMPLE_ENGLISH_QUESTIONS = [
    # Level 1 - Basic Grammar
    {
        "subject": Subject.ENGLISH,
        "content": "Choose the correct form: I ____ to school every day.",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 1,
        "topic_category": "present_tense",
        "correct_answer": "go",
        "options": ["go", "goes", "going", "went"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 30,
            "learning_objectives": ["present tense with 'I'"],
            "tags": ["grammar", "present_tense", "basic"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "What is the plural of 'cat'?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 1,
        "topic_category": "plurals",
        "correct_answer": "cats",
        "options": ["cat", "cats", "cates", "catss"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 20,
            "learning_objectives": ["basic plurals"],
            "tags": ["grammar", "plurals", "basic"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "Fill in the blank: This is ____ apple.",
        "question_type": QuestionType.FILL_BLANK,
        "difficulty_level": 1,
        "topic_category": "articles",
        "correct_answer": "an",
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 30,
            "learning_objectives": ["articles a/an"],
            "tags": ["grammar", "articles", "basic"]
        }
    },
    
    # Level 2 - Past Tense and Basic Vocabulary
    {
        "subject": Subject.ENGLISH,
        "content": "Choose the past tense of 'go':",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 2,
        "topic_category": "past_tense",
        "correct_answer": "went",
        "options": ["goed", "went", "going", "goes"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 45,
            "learning_objectives": ["irregular past tense"],
            "tags": ["grammar", "past_tense", "irregular_verbs"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "Fill in the blank: Yesterday, I _____ (watch) a movie.",
        "question_type": QuestionType.FILL_BLANK,
        "difficulty_level": 2,
        "topic_category": "past_tense",
        "correct_answer": "watched",
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 60,
            "learning_objectives": ["regular past tense formation"],
            "tags": ["grammar", "past_tense", "regular_verbs"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "What does 'happy' mean?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 2,
        "topic_category": "vocabulary",
        "correct_answer": "feeling joy",
        "options": ["feeling sad", "feeling joy", "feeling angry", "feeling tired"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 30,
            "learning_objectives": ["basic emotions vocabulary"],
            "tags": ["vocabulary", "emotions", "adjectives"]
        }
    },
    
    # Level 3 - Present Perfect and Intermediate Grammar
    {
        "subject": Subject.ENGLISH,
        "content": "Choose the correct answer: I ____ never been to Paris.",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 3,
        "topic_category": "present_perfect",
        "correct_answer": "have",
        "options": ["have", "has", "had", "having"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 90,
            "learning_objectives": ["present perfect tense"],
            "tags": ["grammar", "present_perfect", "auxiliary_verbs"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "Fill in the blank: She _____ (live) here for five years.",
        "question_type": QuestionType.FILL_BLANK,
        "difficulty_level": 3,
        "topic_category": "present_perfect",
        "correct_answer": "has lived",
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 120,
            "learning_objectives": ["present perfect with duration"],
            "tags": ["grammar", "present_perfect", "duration"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "Which preposition is correct: I'm interested ____ learning English.",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 3,
        "topic_category": "prepositions",
        "correct_answer": "in",
        "options": ["in", "on", "at", "for"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 75,
            "learning_objectives": ["prepositions with adjectives"],
            "tags": ["grammar", "prepositions", "adjective_patterns"]
        }
    },
    
    # Level 4 - Complex Grammar and Advanced Vocabulary
    {
        "subject": Subject.ENGLISH,
        "content": "Choose the correct conditional: If I ____ rich, I would travel the world.",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 4,
        "topic_category": "conditionals",
        "correct_answer": "were",
        "options": ["am", "was", "were", "will be"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 150,
            "learning_objectives": ["second conditional", "subjunctive mood"],
            "tags": ["grammar", "conditionals", "subjunctive"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "What is the meaning of 'procrastinate'?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 4,
        "topic_category": "vocabulary",
        "correct_answer": "to delay doing something",
        "options": ["to work quickly", "to delay doing something", "to finish early", "to organize tasks"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 90,
            "learning_objectives": ["advanced vocabulary"],
            "tags": ["vocabulary", "advanced", "academic"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "Identify the passive voice: The book ____ by millions of people.",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 4,
        "topic_category": "passive_voice",
        "correct_answer": "was read",
        "options": ["reads", "read", "was read", "is reading"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 120,
            "learning_objectives": ["passive voice formation"],
            "tags": ["grammar", "passive_voice", "verb_forms"]
        }
    },
    
    # Level 5 - Advanced Grammar and Academic English
    {
        "subject": Subject.ENGLISH,
        "content": "Choose the correct subjunctive: It's important that he ____ on time.",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 5,
        "topic_category": "subjunctive",
        "correct_answer": "be",
        "options": ["is", "be", "will be", "was"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 180,
            "learning_objectives": ["subjunctive mood", "formal expressions"],
            "tags": ["grammar", "subjunctive", "formal", "advanced"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "What does 'ubiquitous' mean?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 5,
        "topic_category": "vocabulary",
        "correct_answer": "present everywhere",
        "options": ["very rare", "present everywhere", "extremely large", "completely invisible"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 120,
            "learning_objectives": ["academic vocabulary"],
            "tags": ["vocabulary", "academic", "advanced", "descriptive"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "Analyze this sentence structure: 'Having finished the project, she felt relieved.' What is 'Having finished'?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 5,
        "topic_category": "sentence_structure",
        "correct_answer": "perfect participle",
        "options": ["gerund", "perfect participle", "infinitive", "past tense"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 200,
            "learning_objectives": ["complex sentence analysis", "participles"],
            "tags": ["grammar", "sentence_structure", "participles", "advanced"]
        }
    },
    
    # Additional question types for variety
    {
        "subject": Subject.ENGLISH,
        "content": "True or False: 'Their' and 'there' have the same meaning.",
        "question_type": QuestionType.TRUE_FALSE,
        "difficulty_level": 2,
        "topic_category": "homophones",
        "correct_answer": "False",
        "options": ["True", "False"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 45,
            "learning_objectives": ["homophones distinction"],
            "tags": ["vocabulary", "homophones", "spelling"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "Write a sentence using the word 'although' to show contrast.",
        "question_type": QuestionType.OPEN_ENDED,
        "difficulty_level": 3,
        "topic_category": "conjunctions",
        "correct_answer": "Although it was raining, we went for a walk.",
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 180,
            "learning_objectives": ["contrast conjunctions", "sentence construction"],
            "tags": ["grammar", "conjunctions", "sentence_writing"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "Complete the idiom: 'It's raining cats and ____'",
        "question_type": QuestionType.FILL_BLANK,
        "difficulty_level": 4,
        "topic_category": "idioms",
        "correct_answer": "dogs",
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 60,
            "learning_objectives": ["common English idioms"],
            "tags": ["vocabulary", "idioms", "expressions"]
        }
    },
    {
        "subject": Subject.ENGLISH,
        "content": "Which word is a synonym for 'enormous'?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 3,
        "topic_category": "synonyms",
        "correct_answer": "huge",
        "options": ["tiny", "huge", "medium", "narrow"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 45,
            "learning_objectives": ["synonyms and word relationships"],
            "tags": ["vocabulary", "synonyms", "adjectives"]
        }
    }
]