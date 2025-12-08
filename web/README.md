# OpenXtract Web Application

A full-stack web application for extracting structured data from documents using AI.

## Architecture

- **Frontend**: NextJS 14 with App Router, Tailwind CSS, shadcn/ui
- **Backend**: FastAPI wrapping the open-xtract Python library
- **Database**: PostgreSQL for users, schemas, and extraction history
- **Storage**: MinIO (S3-compatible) for uploaded documents
- **Cache/Queue**: Redis for caching and background jobs

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 20+ (for local frontend development)
- Python 3.12+ (for local backend development)

### Development

1. Start all services with Docker Compose:

```bash
cd web/docker
docker compose up -d
```

2. The services will be available at:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - MinIO Console: http://localhost:9001

3. Run database migrations:

```bash
docker compose exec backend alembic upgrade head
```

### Local Development (without Docker)

#### Backend

```bash
cd web/backend

# Install dependencies
pip install -e .

# Start PostgreSQL and MinIO (via Docker)
docker compose up postgres minio redis -d

# Run migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd web/frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp web/docker/.env.example web/docker/.env
```

Key variables:
- `SECRET_KEY`: JWT signing key
- `ENCRYPTION_KEY`: API key encryption key
- `POSTGRES_PASSWORD`: Database password
- `MINIO_SECRET_KEY`: MinIO secret
- `NEXTAUTH_SECRET`: NextAuth session secret

### OAuth Setup (Optional)

For Google/GitHub login:

1. Create OAuth applications:
   - Google: https://console.cloud.google.com/apis/credentials
   - GitHub: https://github.com/settings/developers

2. Add credentials to `.env`:
   - `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`
   - `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login
- `POST /api/v1/auth/refresh` - Refresh token
- `GET /api/v1/auth/me` - Current user

### Schemas
- `GET /api/v1/schemas` - List schemas
- `POST /api/v1/schemas` - Create schema
- `GET /api/v1/schemas/{id}` - Get schema
- `PUT /api/v1/schemas/{id}` - Update schema
- `DELETE /api/v1/schemas/{id}` - Delete schema

### Extractions
- `GET /api/v1/extractions` - List extractions
- `POST /api/v1/extractions` - Create extraction
- `GET /api/v1/extractions/{id}` - Get extraction
- `POST /api/v1/extractions/upload` - Upload file

### Provider Management
- `GET /api/v1/users/me/providers` - List configured providers
- `POST /api/v1/users/me/providers` - Add provider API key
- `DELETE /api/v1/users/me/providers/{provider}` - Remove provider

## Project Structure

```
web/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI entry point
│   │   ├── config.py         # Settings
│   │   ├── database.py       # SQLAlchemy setup
│   │   ├── dependencies.py   # Auth dependencies
│   │   ├── models/           # SQLAlchemy models
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── routers/          # API routes
│   │   ├── services/         # Business logic
│   │   └── utils/            # Helpers
│   ├── alembic/              # Migrations
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── app/              # NextJS pages
│   │   ├── components/       # React components
│   │   ├── lib/              # API client, auth
│   │   └── hooks/            # Custom hooks
│   ├── Dockerfile
│   └── package.json
└── docker/
    ├── docker-compose.yml      # Development
    ├── docker-compose.prod.yml # Production
    └── nginx/                  # Reverse proxy config
```

## Production Deployment

1. Configure production environment:

```bash
cp web/docker/.env.example web/docker/.env
# Edit .env with production values
```

2. Build and start:

```bash
cd web/docker
docker compose -f docker-compose.prod.yml up -d --build
```

3. Run migrations:

```bash
docker compose -f docker-compose.prod.yml exec backend alembic upgrade head
```
