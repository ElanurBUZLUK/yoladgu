# Schemas package

# Import all schemas for easy access
from .question import *
from .answer import *
from .auth import *
from .dashboard import *
from .spaced_repetition import *
from .pdf_upload import *
from .user import *
from .error import *
from .api_docs import *
from .llm_management import *
from .vector_management import *

__all__ = [
    # Question schemas
    "QuestionBase", "QuestionCreate", "QuestionUpdate", "QuestionResponse",
    "QuestionRecommendationRequest", "QuestionRecommendationResponse", "QuestionSearchQuery",
    "QuestionStats", "QuestionDifficultyAdjustment", "QuestionPool", "QuestionMetadata",
    "BulkQuestionOperation", "QuestionValidationResult",
    
    # Answer schemas
    "AnswerBase", "AnswerCreate", "AnswerResponse", "AnswerEvaluation",
    "AnswerStats", "AnswerSearchQuery", "AnswerBulkOperation",
    
    # Auth schemas
    "UserLogin", "UserLoginResponse", "UserRefreshToken", "UserPasswordChange",
    "UserPasswordReset", "UserPasswordResetConfirm",
    
    # Dashboard schemas
    "DashboardData", "DashboardStats", "DashboardChart", "DashboardWidget",
    
    # Spaced repetition schemas
    "SpacedRepetitionData", "ReviewSchedule", "ReviewSession", "ReviewResult",
    
    # PDF upload schemas
    "PDFUploadCreate", "PDFUploadResponse", "PDFUploadStatus", "PDFProcessingResult",
    
    # User schemas
    "UserBase", "UserCreate", "UserUpdate", "UserResponse", "UserLogin", "UserLoginResponse",
    "UserRefreshToken", "UserPasswordChange", "UserPasswordReset", "UserPasswordResetConfirm",
    "UserLevelUpdate", "UserSearchQuery", "UserStats", "UserSummary", "UserBulkOperation",
    "UserCountByRole", "UserStatsSummary", "UserValidationResult", "UserProfile",
    "UserActivity", "UserAchievement", "UserPreferences", "UserNotification",
    
    # Error schemas
    "ErrorResponse", "ValidationErrorResponse", "AuthenticationErrorResponse",
    "AuthorizationErrorResponse", "NotFoundErrorResponse", "ConflictErrorResponse",
    "RateLimitErrorResponse", "InternalErrorResponse", "ExternalServiceErrorResponse",
    "DatabaseErrorResponse", "FileUploadErrorResponse", "LLMErrorResponse", "MCPErrorResponse",
    "ErrorType", "ErrorSeverity", "ErrorDetail", "ErrorLog", "ErrorSummary", "ErrorMetrics",
    "ErrorAlert", "ErrorRecoveryAction", "ErrorPreventionRule",
    
    # API documentation schemas
    "ApiDocumentation", "ApiEndpoint", "ApiGroup", "ApiVersion", "ApiExample",
    "ApiTutorial", "ApiChangelog", "ApiMetric", "ApiHealthCheck", "ApiRateLimit",
    "ApiUsage", "ApiSchema", "ApiErrorCode", "ApiSDK",

    # LLM Management schemas
    "PolicySelectionRequest", "PolicySelectionResponse",
    "CostLimitRequest", "CostLimitResponse",
    "ContentModerationRequest", "ContentModerationResponse",
    "LLMHealthRequest", "LLMHealthResponse",
    "GetAllPoliciesResponse", "UsageReportEntry", "GetUsageReportResponse",
    "ModerationStatsEntry", "GetModerationStatsResponse",
    "CheckUserFlagStatusResponse", "ProviderStatusEntry", "GetProviderStatusResponse",
    "TestPolicySelectionResponse",

    # Vector Management schemas
    "BatchUpsertRequest", "BatchUpsertResponse",
    "RebuildIndexRequest", "RebuildIndexResponse",
    "CleanupRequest", "CleanupResponse",
    "LockStatusResponse",
    "GetVectorStatisticsResponse", "ForceReleaseLockResponse",
    "RebuildIndexManualResponse", "VectorHealthCheckResponse"
]