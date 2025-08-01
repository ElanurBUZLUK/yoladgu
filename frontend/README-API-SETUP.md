# 🔧 Frontend API Configuration Guide

## 📋 Overview
This frontend is configured to properly communicate with the FastAPI backend running on `localhost:8000`.

## 🚀 Quick Start

### Development Mode
```bash
# 1. Start Backend (Terminal 1)
cd backend
python run.py

# 2. Start Frontend (Terminal 2)  
cd frontend
npm run start

# Frontend runs on: http://localhost:4200
# Backend runs on: http://localhost:8000
```

## 📡 API Configuration

### Environment Files
- **Development**: `src/environments/environment.ts`
  - Uses proxy: `/api/v1` → `http://localhost:8000/api/v1`
- **Production**: `src/environments/environment.prod.ts`
  - Direct URL: `http://127.0.0.1:8000/api/v1`

### Proxy Configuration
File: `proxy.conf.json`
```json
{
  "/api/*": {
    "target": "http://localhost:8000",
    "secure": false,
    "changeOrigin": true,
    "logLevel": "debug"
  }
}
```

### API Config Class
Central API URL management in `src/app/core/config/api.config.ts`:
- Handles development vs production URLs
- Provides debugging helpers
- Environment-aware URL generation

## 🔍 Debugging

### API Debug Component
- Click "🔧 Debug API" button (bottom-right)
- Shows current configuration
- Test API connectivity
- Troubleshoot connection issues

### Console Logs
Services log their API configuration on startup:
```
QuestionService API Config: {production: false, apiUrl: "/api/v1", ...}
AuthService API Config: {production: false, apiUrl: "/api/v1", ...}
StudentService API Config: {production: false, apiUrl: "/api/v1", ...}
```

## 🌐 HTTP Interceptors

### 1. ApiBaseUrlInterceptor
- Handles URL routing (development vs production)
- Logs API requests for debugging

### 2. TokenInterceptor  
- Adds JWT authentication headers
- Reads token from localStorage

## 📊 Service Endpoints

### QuestionService
- `GET /api/v1/recommendations/next-question` - Get next question
- `POST /api/v1/questions/{id}/answer` - Submit answer
- `POST /api/v1/quiz-sessions` - Submit quiz results
- `PUT /api/v1/users/me/progress` - Update progress

### AuthService
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration
- `GET /api/v1/users/me` - Get current user

### StudentService
- `GET /api/v1/users/me` - Get profile
- `GET /api/v1/users/me/level` - Get level
- `GET /api/v1/subjects/` - Get subjects
- `GET /api/v1/analytics/student-analytics` - Get analytics

## 🔧 Troubleshooting

### Common Issues

#### 1. "Cannot reach backend"
**Solution**: Make sure backend is running:
```bash
cd backend
python run.py
```

#### 2. "404 Not Found" errors
**Solution**: Check if backend endpoints exist:
```bash
curl http://localhost:8000/api/v1/health
```

#### 3. "CORS errors"
**Solution**: Backend should have CORS enabled for `localhost:4200`

#### 4. "Proxy not working"
**Solution**: Ensure you're using the correct start command:
```bash
npm run start  # Uses proxy configuration
```

### Network Debugging
```bash
# Check if backend is running
curl http://localhost:8000/api/v1/

# Check proxy in development
# Frontend requests to /api/v1/* should be routed to localhost:8000/api/v1/*
```

## 📝 Notes

1. **Development**: Uses Angular proxy for seamless API calls
2. **Production**: Uses direct backend URLs
3. **Authentication**: JWT tokens stored in localStorage
4. **Error Handling**: Global error management with toast notifications
5. **Real-time**: Quiz data syncs to backend automatically

## 🚀 Production Deployment

For production, ensure:
1. Update `environment.prod.ts` with correct backend URL
2. Build with: `npm run build`
3. Serve static files from `dist/` directory
4. Configure server to handle Angular routing (fallback to index.html)