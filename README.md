# Adaptive Learning Platform

Modern, AI-powered adaptive learning platform with personalized question selection and intelligent feedback.

## 🚀 Features

- **Adaptive Question Selection**: AI-powered question selection based on student performance
- **Math & English Support**: Comprehensive coverage of both subjects
- **Vector Search**: Advanced similarity search using pgvector
- **Personalization**: Learning style adaptation and personalized recommendations
- **Analytics**: Detailed performance analytics and insights
- **MCP Integration**: Model Context Protocol for advanced AI capabilities
- **Real-time Monitoring**: System health monitoring and performance metrics

## 🏗️ Architecture

```
backend/
├── app/
│   ├── api/v1/           # API endpoints
│   ├── core/             # Core configuration and utilities
│   ├── models/           # Database models
│   ├── services/         # Business logic services
│   ├── middleware/       # Request/response middleware
│   └── mcp/             # Model Context Protocol integration
├── alembic/             # Database migrations
├── scripts/             # Setup and utility scripts
└── tests/               # Test files
```

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL 13+ with pgvector extension
- Redis 6+
- Node.js 16+ (for frontend)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd yoladgunew
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp env.example .env

# Edit .env file with your configuration
nano .env
```

### 3. Database Setup

```bash
# Install pgvector extension using automated script
chmod +x scripts/install_pgvector.sh
./scripts/install_pgvector.sh

# Or install manually:
# Ubuntu/Debian: sudo apt-get install postgresql-13-pgvector
# CentOS/RHEL: sudo yum install pgvector_13
# macOS: brew install pgvector

# Or using Docker
docker run --name postgres-pgvector -e POSTGRES_PASSWORD=password -e POSTGRES_DB=adaptive_learning -p 5432:5432 -d pgvector/pgvector:pg13

# Run database migrations (includes pgvector extension and improved indexes)
alembic upgrade head
```

### 4. Redis Setup

```bash
# Install Redis (Ubuntu/Debian)
sudo apt-get install redis-server

# Or using Docker
docker run --name redis -p 6379:6379 -d redis:6-alpine
```

### 5. System Setup

```bash
# Run the setup script
python scripts/setup_system.py
```

### 6. Start the Application

```bash
# Development mode
python run_dev.py

# Or using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/adaptive_learning

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-super-secret-key-change-in-production
JWT_SECRET=your-jwt-secret-change-in-production

# LLM Providers
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Vector Database
PGVECTOR_ENABLED=true
VECTOR_SIMILARITY_THRESHOLD=0.7
```

### Required Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: Application secret key
- `JWT_SECRET`: JWT signing secret

### Optional Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for LLM features
- `ANTHROPIC_API_KEY`: Anthropic API key for LLM features
- `PGVECTOR_ENABLED`: Enable pgvector features (default: true)

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_math_api.py

# Run with coverage
pytest --cov=app tests/
```

### Test API Endpoints

```bash
# Test math RAG endpoints
python x/test_math_api.py

# Test English RAG endpoints
python x/test_english_api.py

# Test authentication
python x/test_auth.py
```

## 📊 API Documentation

Once the application is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🔍 Key Endpoints

### Authentication
- `POST /api/v1/users/register` - User registration
- `POST /api/v1/users/login` - User login
- `POST /api/v1/auth/refresh` - Refresh access token

### Math RAG
- `POST /api/v1/math/rag/next-question` - Get next math question
- `POST /api/v1/math/rag/submit-answer` - Submit math answer
- `GET /api/v1/math/rag/profile` - Get math profile
- `GET /api/v1/math/rag/analytics/learning-progress` - Learning progress analytics

### English RAG
- `POST /api/v1/english/rag/next-question` - Get next English question
- `POST /api/v1/english/rag/submit-answer` - Submit English answer
- `GET /api/v1/english/rag/profile` - Get English profile

### Dashboard
- `GET /api/v1/dashboard/data` - Get dashboard data
- `GET /api/v1/dashboard/subject-selection` - Get subject selection data

## 🏥 Health Checks

### System Health
```bash
curl http://localhost:8000/health
```

### Vector Index Health
```bash
curl http://localhost:8000/api/v1/math/rag/monitoring/system-health
```

### Database Health
```bash
curl http://localhost:8000/api/v1/system/health
```

## 🔧 Troubleshooting

### Common Issues

1. **PgVector Extension Not Found**
   ```bash
   # Install pgvector extension
   sudo apt-get install postgresql-13-pgvector
   ```

2. **Database Connection Failed**
   - Check `DATABASE_URL` in `.env`
   - Ensure PostgreSQL is running
   - Verify database exists

3. **Redis Connection Failed**
   - Check `REDIS_URL` in `.env`
   - Ensure Redis is running

4. **Vector Index Creation Failed**
   - Ensure pgvector extension is installed
   - Check database permissions
   - Run setup script: `python scripts/setup_system.py`

### Logs

Check application logs for detailed error information:

```bash
# View application logs
tail -f logs/app.log

# View error logs
tail -f logs/error.log
```

## 📈 Performance

### Optimization Tips

1. **Database Indexes**: Ensure all necessary indexes are created
2. **Vector Search**: Use appropriate similarity thresholds
3. **Caching**: Leverage Redis for frequently accessed data
4. **Batch Processing**: Use batch operations for embeddings

### Monitoring

- **Performance Metrics**: `/api/v1/math/rag/monitoring/performance-metrics`
- **System Health**: `/api/v1/math/rag/monitoring/system-health`
- **Alerts**: `/api/v1/math/rag/monitoring/alerts`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

1. Check the [documentation](docs/)
2. Search existing [issues](issues/)
3. Create a new issue with detailed information

## 🗺️ Roadmap

- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Mobile app
- [ ] Integration with LMS platforms
- [ ] Advanced personalization algorithms
- [ ] Real-time collaboration features
