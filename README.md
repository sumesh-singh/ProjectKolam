# Kolam Design Pattern Recognition and Recreation System

A full-stack web application that uses AI and computer vision to analyze traditional Indian Kolam designs, extract their mathematical properties, and recreate authentic patterns while preserving cultural heritage.

## 🌟 Features

### 🔐 **Authentication & User Management**

- User registration and login with JWT authentication
- Role-based access control (User, Expert, Admin)
- Secure password hashing and session management
- Protected routes and authentication guards

### 🎨 **Pattern Recognition & Analysis**

- Upload and analyze Kolam images using advanced computer vision
- Automatic classification by type (Kolam, Muggu, Rangoli, Rangavalli)
- Mathematical property extraction (symmetry, geometry, complexity)
- Cultural context identification and metadata enrichment

### 🛠️ **Interactive Design Tools**

- Real-time pattern recreation based on mathematical principles
- Parameterized design generation (size, complexity, style)
- Traditional rule-based pattern creation
- Visual preview and editing capabilities

### 📚 **Cultural Documentation**

- Comprehensive database of traditional designs
- Multi-language support for cultural content
- Historical and ceremonial context preservation
- Expert validation workflows

### 🖼️ **Gallery & Community**

- Browse and search traditional Kolam patterns
- User-generated content and pattern sharing
- Cultural insights and educational content
- Community-driven validation and feedback

## 🏗️ **Architecture**

### Backend (FastAPI)

- **Framework**: FastAPI with async support
- **Database**: PostgreSQL (structured data) + MongoDB (documents/images) + Redis (caching)
- **Authentication**: JWT tokens with role-based permissions
- **ML/AI**: TensorFlow, PyTorch, OpenCV for pattern analysis
- **API**: RESTful endpoints with automatic OpenAPI documentation

### Frontend (React + TypeScript)

- **Framework**: React 19 with TypeScript
- **State Management**: Zustand for client-side state
- **Styling**: Tailwind CSS with custom design system
- **Routing**: React Router with protected routes
- **HTTP Client**: Axios with interceptors for auth

### DevOps & Deployment

- **Containerization**: Docker with multi-service setup
- **Orchestration**: Docker Compose for local development
- **CI/CD Ready**: GitHub Actions compatible
- **Monitoring**: Health checks and structured logging

## 🚀 **Quick Start**

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ and npm
- Python 3.11+ (for local development)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd kolam-design-system
```

### 2. Start Backend Services

```bash
cd backend
docker-compose up -d
```

### 3. Install Frontend Dependencies

```bash
npm install
```

### 4. Start Frontend Development Server

```bash
npm run dev
```

### 5. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📁 **Project Structure**

```
kolam-design-system/
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── api/v1/endpoints/        # API route handlers
│   │   ├── core/                    # Core functionality
│   │   ├── crud/                    # Database operations
│   │   ├── db/                      # Database connections
│   │   ├── models/                  # SQLAlchemy models
│   │   └── schemas/                 # Pydantic schemas
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # Backend container
│   └── docker-compose.yml           # Multi-service setup
├── src/                             # React Frontend
│   ├── components/                  # Reusable components
│   ├── pages/                       # Page components
│   ├── services/                    # API services
│   ├── store/                       # Zustand stores
│   └── ...
├── public/                          # Static assets
├── package.json                     # Frontend dependencies
├── .env                             # Environment variables
└── README.md                        # This file
```

## 🔧 **Configuration**

### Environment Variables

**Frontend (.env)**:

```env
VITE_API_URL=http://localhost:8000/api/v1
```

**Backend (backend/.env)**:

```env
DEBUG=1
SECRET_KEY=your-secret-key-change-in-production
DATABASE_URL=postgresql://user:password@localhost/db
MONGODB_URL=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379
```

## 🛠️ **Development**

### Backend Development

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### Frontend Development

```bash
npm run dev          # Start dev server
npm run build        # Build for production
npm run lint         # Run ESLint
npm run preview      # Preview production build
```

### Database Setup

```bash
cd backend
docker-compose up -d postgres mongodb redis
alembic upgrade head  # Run database migrations
```

## 🔒 **Security Features**

- **Authentication**: JWT-based with secure token storage
- **Authorization**: Role-based access control
- **Data Protection**: Input validation and sanitization
- **API Security**: Rate limiting and CORS protection
- **Password Security**: bcrypt hashing with strength requirements

## 📊 **API Endpoints**

### Authentication

- `POST /api/v1/users/register` - User registration
- `POST /api/v1/users/login` - User authentication
- `GET /api/v1/users/me` - Current user profile

### Pattern Analysis

- `POST /api/v1/patterns/analyze` - Analyze uploaded image
- `GET /api/v1/patterns/analysis/{id}` - Get analysis results

### Design Management

- `GET /api/v1/designs` - List designs with filtering
- `POST /api/v1/designs` - Create new design
- `GET /api/v1/designs/{id}` - Get design details

## 🎨 **Design System**

The application uses a custom design system with:

- **Colors**: Primary red, gold accents, indigo highlights
- **Typography**: Serif for headings, sans-serif for body text
- **Components**: Consistent button styles, form elements, cards
- **Responsive**: Mobile-first design with Tailwind CSS

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow TypeScript and Python type hints
- Write tests for new features
- Update documentation
- Use conventional commit messages
- Ensure responsive design

## 📈 **Performance**

- **Frontend**: Code splitting, lazy loading, optimized images
- **Backend**: Async operations, database indexing, caching
- **API**: Pagination, efficient queries, response compression
- **Monitoring**: Health checks, error tracking, performance metrics

## 🧪 **Testing**

### Backend Tests

```bash
cd backend
pytest
```

### Frontend Tests

```bash
npm test
```

## 🚀 **Deployment**

### Production Build

```bash
# Build frontend
npm run build

# Build and deploy backend
cd backend
docker build -t kolam-backend .
docker-compose -f docker-compose.prod.yml up -d
```

### Environment Setup

- Configure production database URLs
- Set secure SECRET_KEY
- Enable HTTPS and SSL certificates
- Configure monitoring and logging

## 📄 **License**

This project is developed for the Smart India Hackathon 2025.

## 🙏 **Cultural Respect**

This application is built with deep respect for Indian cultural traditions. All pattern analysis and recreation features are designed to preserve and celebrate the mathematical beauty and cultural significance of Kolam art forms.

## 📞 **Support**

For questions or support:

- Create an issue in the repository
- Contact the development team
- Check the API documentation at `/docs`

---

**Built with ❤️ for preserving cultural heritage through technology**
