# Kolam Design System Backend

A comprehensive backend API for the Kolam Design Pattern Recognition and Recreation System, built with FastAPI.

## Features

- **User Management**: Registration, authentication, and authorization with JWT tokens
- **Pattern Recognition**: AI-powered analysis of Kolam designs using computer vision and ML
- **Design Recreation**: Algorithmic generation of authentic Kolam patterns
- **Cultural Documentation**: Rich metadata and cultural context for designs
- **Scalable Architecture**: Microservices-ready with PostgreSQL, MongoDB, and Redis

## Technology Stack

- **Framework**: FastAPI with async support
- **Databases**:
  - PostgreSQL: Structured data (users, designs, metadata)
  - MongoDB: Document storage (images, analysis data)
  - Redis: Caching and session management
- **Authentication**: JWT tokens with role-based access control
- **ML/AI**: TensorFlow, PyTorch, OpenCV, scikit-learn
- **Deployment**: Docker, Kubernetes-ready

## Quick Start

### Using Docker Compose (Recommended)

1. **Start all services**:

   ```bash
   cd backend
   docker-compose up -d
   ```

2. **Access the API**:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Local Development

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up databases**:

   - PostgreSQL on port 5432
   - MongoDB on port 27017
   - Redis on port 6379

3. **Run the application**:
   ```bash
   python -m uvicorn app.main:app --reload
   ```

## API Endpoints

### Authentication

- `POST /api/v1/users/register` - User registration
- `POST /api/v1/users/login` - User login
- `GET /api/v1/users/me` - Get current user

### Pattern Analysis

- `POST /api/v1/patterns/analyze` - Analyze uploaded Kolam image
- `GET /api/v1/patterns/analysis/{analysis_id}` - Get analysis results

## Project Structure

```
backend/
├── app/
│   ├── api/v1/
│   │   ├── endpoints/     # API route handlers
│   │   └── api.py         # Main API router
│   ├── core/              # Core functionality
│   │   ├── auth.py        # Authentication dependencies
│   │   ├── config.py      # Settings and configuration
│   │   ├── logging.py     # Logging setup
│   │   └── security.py    # Security utilities
│   ├── crud/              # Database operations
│   ├── db/                # Database connections
│   ├── models/            # SQLAlchemy models
│   ├── schemas/           # Pydantic schemas
│   └── main.py            # FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Multi-service setup
└── .env                  # Environment variables
```

## Environment Variables

Key configuration options in `.env`:

```env
DEBUG=1
SECRET_KEY=your-secret-key-change-in-production
DATABASE_URL=postgresql://user:password@localhost/db
MONGODB_URL=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
isort .
```

### API Documentation

- Interactive docs: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json

## Deployment

### Production Docker Build

```bash
docker build -t kolam-backend .
docker run -p 8000:8000 kolam-backend
```

### Kubernetes Deployment

The application is designed to work with Kubernetes orchestration. See the deployment manifests in the `k8s/` directory (to be created).

## Security Features

- JWT-based authentication
- Role-based access control (user, expert, admin)
- Input validation and sanitization
- Rate limiting
- CORS protection
- SQL injection prevention
- Secure password hashing

## Performance

- Async database operations
- Redis caching for frequently accessed data
- Connection pooling
- Optimized queries with proper indexing
- Background task processing with Celery

## Monitoring

- Health check endpoints
- Structured logging with JSON output
- Prometheus metrics (configurable)
- Request/response logging middleware

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Use type hints and follow PEP 8
5. Create feature branches for development

## License

This project is part of the Smart India Hackathon 2025 submission.
