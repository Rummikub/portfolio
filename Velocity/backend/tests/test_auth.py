import pytest
import json
from bson import ObjectId

@pytest.mark.asyncio
async def test_register_user(client, test_db):
    """Test user registration"""
    workspace_id = str(ObjectId())
    
    user_data = {
        "email": "newuser@example.com",
        "password": "testpass123",
        "first_name": "New",
        "last_name": "User"
    }
    
    response = client.post('/api/auth/register', 
                          json=user_data,
                          headers={
                              'X-Workspace-Id': workspace_id,
                              'Content-Type': 'application/json'
                          })
    
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data['ok'] is True
    assert 'data' in data
    assert data['data']['email'] == user_data['email']
    assert data['data']['first_name'] == user_data['first_name']

@pytest.mark.asyncio 
async def test_register_duplicate_email(client, test_db, auth_headers):
    """Test registering with duplicate email"""
    workspace_id = auth_headers['X-Workspace-Id']
    
    user_data = {
        "email": "test@example.com",  # This email is already used in auth_headers fixture
        "password": "testpass123",
        "first_name": "Duplicate",
        "last_name": "User"
    }
    
    response = client.post('/api/auth/register',
                          json=user_data,
                          headers={
                              'X-Workspace-Id': workspace_id,
                              'Content-Type': 'application/json'
                          })
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['ok'] is False
    assert 'error' in data

@pytest.mark.asyncio
async def test_login_success(client, test_db):
    """Test successful login"""
    workspace_id = str(ObjectId())
    
    # First register a user
    user_data = {
        "email": "logintest@example.com",
        "password": "testpass123",
        "first_name": "Login",
        "last_name": "Test"
    }
    
    client.post('/api/auth/register',
                json=user_data,
                headers={
                    'X-Workspace-Id': workspace_id,
                    'Content-Type': 'application/json'
                })
    
    # Then try to login
    login_data = {
        "email": "logintest@example.com",
        "password": "testpass123"
    }
    
    response = client.post('/api/auth/login',
                          json=login_data,
                          headers={'Content-Type': 'application/json'})
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['ok'] is True
    assert 'data' in data
    assert 'access_token' in data['data']
    assert 'refresh_token' in data['data']
    assert 'user' in data['data']

@pytest.mark.asyncio
async def test_login_invalid_credentials(client):
    """Test login with invalid credentials"""
    login_data = {
        "email": "nonexistent@example.com",
        "password": "wrongpass"
    }
    
    response = client.post('/api/auth/login',
                          json=login_data,
                          headers={'Content-Type': 'application/json'})
    
    assert response.status_code == 401
    data = json.loads(response.data)
    assert data['ok'] is False
    assert 'error' in data

@pytest.mark.asyncio
async def test_get_current_user(client, auth_headers):
    """Test getting current user profile"""
    response = client.get('/api/auth/me', headers=auth_headers)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['ok'] is True
    assert 'data' in data
    assert data['data']['email'] == 'test@example.com'

@pytest.mark.asyncio
async def test_get_current_user_unauthorized(client):
    """Test getting current user without auth"""
    response = client.get('/api/auth/me')
    
    assert response.status_code == 401
    data = json.loads(response.data)
    assert data['ok'] is False
    assert 'error' in data

@pytest.mark.asyncio
async def test_logout(client, auth_headers):
    """Test logout endpoint"""
    response = client.post('/api/auth/logout', headers=auth_headers)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['ok'] is True