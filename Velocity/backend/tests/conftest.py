import pytest
import asyncio
from app import create_app
from app.core.database import get_db, db_client
from app.utils.jobs import init_job_queue

@pytest.fixture
def app():
    """Create test app"""
    app = create_app('testing')
    
    # Initialize job queue for testing
    init_job_queue('memory')
    
    return app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture
async def async_app():
    """Create async test app"""
    app = create_app('testing')
    
    with app.app_context():
        yield app

@pytest.fixture
async def test_db(async_app):
    """Create test database and clean up after"""
    db = get_db()
    
    # Clean up test data before test
    await db.users.delete_many({})
    await db.agents.delete_many({})
    await db.docs.delete_many({})
    await db.design_partners.delete_many({})
    await db.research.delete_many({})
    
    yield db
    
    # Clean up test data after test
    await db.users.delete_many({})
    await db.agents.delete_many({})
    await db.docs.delete_many({})
    await db.design_partners.delete_many({})
    await db.research.delete_many({})

@pytest.fixture
async def auth_headers(test_db):
    """Create test user and return auth headers"""
    from app.core.auth_service import AuthService
    from app.schemas.base import CreateUserRequest
    from bson import ObjectId
    from flask_jwt_extended import create_access_token
    
    # Create test workspace
    workspace_id = str(ObjectId())
    
    # Create test user
    user_data = CreateUserRequest(
        email="test@example.com",
        password="testpass123",
        first_name="Test",
        last_name="User"
    )
    
    user = await AuthService.create_user(user_data, workspace_id)
    
    # Create access token
    access_token = create_access_token(identity=user.id)
    
    return {
        'Authorization': f'Bearer {access_token}',
        'X-Workspace-Id': workspace_id,
        'Content-Type': 'application/json'
    }

@pytest.fixture
def sample_agent_data():
    """Sample agent data for testing"""
    return {
        "name": "Test Agent",
        "description": "A test agent for market research",
        "type": "market_discovery",
        "config": {
            "timeout": 300,
            "max_results": 100
        }
    }

@pytest.fixture
def sample_doc_data():
    """Sample document data for testing"""
    return {
        "title": "Test Document",
        "content": "This is a test document content",
        "type": "research",
        "tags": ["test", "research"],
        "metadata": {
            "source": "test",
            "version": "1.0"
        }
    }

@pytest.fixture
def sample_partner_data():
    """Sample design partner data for testing"""
    return {
        "name": "Test Company",
        "email": "contact@testcompany.com",
        "company": "Test Company Inc",
        "industry": "Technology",
        "status": "prospect",
        "notes": "Initial contact made"
    }

@pytest.fixture
def sample_research_data():
    """Sample research data for testing"""
    return {
        "title": "Market Analysis Test",
        "type": "market_analysis",
        "data": {
            "market_size": 1000000,
            "growth_rate": 15.5,
            "key_players": ["Company A", "Company B"]
        },
        "source": "internal_research",
        "confidence_score": 0.85,
        "tags": ["market", "analysis", "test"]
    }