# Agentic System Backend

A Flask-based backend starter for an agentic system that performs market discovery, validation, customer finding, and solution testing through automated agents.

## Features

- **Modular Architecture**: Flask app factory pattern with organized blueprints
- **Authentication**: JWT-based auth with access/refresh tokens and RBAC
- **Database**: MongoDB with Motor async driver and workspace scoping
- **Validation**: Pydantic schemas for request/response validation
- **Error Handling**: Consistent JSON error responses
- **Testing**: Pytest with async support
- **File Storage**: Abstracted storage with local/S3 support
- **Background Jobs**: Pluggable job queue system
- **CORS**: Configurable CORS for development

## Project Structure

```
backend/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── auth/                    # Authentication blueprint
│   ├── agents/                  # Agent management blueprint
│   ├── docs/                    # Document management blueprint
│   ├── design_partners/         # Design partner management blueprint
│   ├── research/                # Research data management blueprint
│   ├── core/                    # Core utilities
│   │   ├── config.py           # Configuration management
│   │   ├── database.py         # MongoDB connection
│   │   ├── errors.py           # Error handling
│   │   └── auth_service.py     # Authentication service
│   ├── models/                  # Pydantic models
│   ├── schemas/                 # Request/response schemas
│   └── utils/                   # Utility functions
├── tests/                       # Test suite
├── uploads/                     # File uploads (local storage)
├── app.py                      # Application entry point
├── requirements.txt            # Python dependencies
└── .env.example               # Environment variables template
```

## Quick Start

### Prerequisites

- Python 3.8+
- MongoDB
- Redis (optional, for Celery)

### Installation

1. **Clone and setup**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start MongoDB** (if using local instance):
```bash
mongod --dbpath /your/db/path
```

4. **Run the application**:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Environment Variables

Key environment variables (see `.env.example` for full list):

- `MONGO_URI`: MongoDB connection string
- `MONGO_DB_NAME`: Database name
- `JWT_SECRET_KEY`: Secret key for JWT tokens
- `CORS_ORIGINS`: Allowed CORS origins (comma-separated)

## API Overview

### Authentication Endpoints

- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `POST /api/auth/refresh` - Refresh access token
- `GET /api/auth/me` - Get current user profile
- `POST /api/auth/logout` - Logout user

### Agent Endpoints

- `POST /api/agents` - Create agent
- `GET /api/agents` - List agents (with pagination/filtering)
- `GET /api/agents/{id}` - Get specific agent
- `PUT /api/agents/{id}` - Update agent
- `DELETE /api/agents/{id}` - Delete agent
- `POST /api/agents/{id}/run` - Run agent (async)
- `POST /api/agents/{id}/stop` - Stop running agent

### Other Endpoints

Similar CRUD patterns for:
- `/api/docs` - Document management
- `/api/design-partners` - Design partner management
- `/api/research` - Research data management

### Authentication

All endpoints (except auth) require:

1. **Authorization header**: `Authorization: Bearer <access_token>`
2. **Workspace header**: `X-Workspace-Id: <workspace_id>`

### Request/Response Format

**Success Response**:
```json
{
  \"ok\": true,
  \"data\": { ... },
  \"message\": \"Optional message\"
}
```

**Error Response**:
```json
{
  \"ok\": false,
  \"error\": {
    \"code\": \"ERROR_CODE\",
    \"message\": \"Error description\",
    \"details\": { ... }
  }
}
```

## Testing

Run tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app
```

## Development

### Adding New Endpoints

1. Create blueprint in `app/{blueprint_name}/`
2. Define routes in `routes.py`
3. Add Pydantic schemas in `app/schemas/`
4. Register blueprint in `app/__init__.py`
5. Add tests in `tests/test_{blueprint_name}.py`

### Database Indexes

The system automatically creates indexes for optimal query performance:
- User email uniqueness
- Workspace + created_at for pagination
- Additional indexes per collection type

### Background Jobs

The job system supports both in-memory (development) and Celery (production) backends:

```python
from app.utils.jobs import enqueue_agent_run

job_id = await enqueue_agent_run(agent_id, config)
```

### File Storage

Switch between local and S3 storage by configuring the storage backend:

```python
from app.utils.storage import create_storage_manager

storage = create_storage_manager()
file_path = await storage.save_file(file_data, filename, workspace_id)
```

## Architecture Notes

### Workspace Scoping

All data is scoped to workspaces via the `X-Workspace-Id` header. This enables:
- Multi-tenant data isolation
- Per-workspace analytics
- Scalable access control

### Agent System Integration

The backend provides endpoints for agent management with hooks for:
- **Market Discovery**: Research market opportunities
- **Validation**: Test assumptions and hypotheses  
- **Customer Finding**: Identify potential design partners
- **Solution Testing**: Run synthetic and real-world tests
- **Deployment**: Manage solution iterations

Agents run as background jobs and report results back to the system.

### Security

- JWT tokens with configurable expiration
- Password hashing with bcrypt
- Role-based access control (user/admin)
- Input validation with Pydantic
- Workspace-level data isolation

## Deployment

### Production Setup

1. Set environment to production in `.env`
2. Use a proper WSGI server (gunicorn, uWSGI)
3. Configure MongoDB with authentication
4. Set up Redis for Celery (if using background jobs)
5. Configure S3 for file storage
6. Set up proper logging and monitoring

### Docker (Optional)

```dockerfile
# Example Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD [\"gunicorn\", \"-w\", \"4\", \"-b\", \"0.0.0.0:5000\", \"app:app\"]
```

## Contributing

1. Follow the existing code patterns
2. Add tests for new features
3. Use type hints consistently
4. Follow PEP 8 style guidelines
5. Update documentation as needed

## License

MIT License