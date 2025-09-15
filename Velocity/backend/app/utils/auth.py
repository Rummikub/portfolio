import bcrypt
from typing import Optional
from flask import request, current_app
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request
from functools import wraps
from app.core.errors import AuthenticationError, AuthorizationError, WorkspaceError
from app.models.base import UserRole
from bson import ObjectId

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def get_current_user_id() -> str:
    """Get current user ID from JWT token"""
    try:
        verify_jwt_in_request()
        user_id = get_jwt_identity()
        if not user_id:
            raise AuthenticationError("Invalid token")
        return user_id
    except Exception as e:
        raise AuthenticationError("Authentication required")

def get_workspace_id() -> str:
    """Get workspace ID from request headers"""
    workspace_id = request.headers.get('X-Workspace-Id')
    if not workspace_id:
        raise WorkspaceError("X-Workspace-Id header is required")
    
    if not ObjectId.is_valid(workspace_id):
        raise WorkspaceError("Invalid workspace ID format")
    
    return workspace_id

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        get_current_user_id()  # This will raise if not authenticated
        return f(*args, **kwargs)
    return wrapper

def require_admin(f):
    """Decorator to require admin role"""
    @wraps(f)
    async def wrapper(*args, **kwargs):
        from app.core.database import get_collection
        
        user_id = get_current_user_id()
        users_collection = get_collection('users')
        
        user = await users_collection.find_one({'_id': ObjectId(user_id)})
        if not user or user.get('role') != UserRole.ADMIN:
            raise AuthorizationError("Admin role required")
        
        return await f(*args, **kwargs)
    return wrapper

def require_workspace_access(f):
    """Decorator to require workspace access"""
    @wraps(f)
    async def wrapper(*args, **kwargs):
        from app.core.database import get_collection
        
        user_id = get_current_user_id()
        workspace_id = get_workspace_id()
        
        # For now, we'll assume users can access any workspace they have the ID for
        # In a real implementation, you'd check workspace membership
        users_collection = get_collection('users')
        user = await users_collection.find_one({'_id': ObjectId(user_id)})
        
        if not user or not user.get('is_active'):
            raise AuthorizationError("User account is inactive")
        
        # Add workspace_id to kwargs for convenience
        kwargs['workspace_id'] = workspace_id
        kwargs['current_user_id'] = user_id
        
        return await f(*args, **kwargs)
    return wrapper