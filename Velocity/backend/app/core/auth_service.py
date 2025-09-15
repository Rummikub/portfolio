from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from flask_jwt_extended import create_access_token, create_refresh_token
from bson import ObjectId
from app.core.database import get_collection
from app.models.base import UserModel, UserRole
from app.schemas.base import CreateUserRequest, LoginRequest, UserSchema
from app.utils.auth import hash_password, verify_password
from app.utils.pagination import convert_objectids_to_strings
from app.core.errors import AuthenticationError, ValidationError

class AuthService:
    """Authentication service for user management"""
    
    @staticmethod
    async def create_user(user_data: CreateUserRequest, workspace_id: str) -> UserSchema:
        """Create a new user"""
        users_collection = get_collection('users')
        
        # Check if user already exists
        existing_user = await users_collection.find_one({"email": user_data.email})
        if existing_user:
            raise ValidationError("User with this email already exists")
        
        # Hash password
        password_hash = hash_password(user_data.password)
        
        # Create user model
        user_model = UserModel(
            workspace_id=ObjectId(workspace_id),
            email=user_data.email,
            password_hash=password_hash,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            role=user_data.role
        )
        
        # Insert user
        result = await users_collection.insert_one(user_model.dict(by_alias=True))
        
        # Fetch created user
        created_user = await users_collection.find_one({"_id": result.inserted_id})
        user_dict = convert_objectids_to_strings(created_user)
        
        return UserSchema(**user_dict)
    
    @staticmethod
    async def authenticate_user(login_data: LoginRequest) -> Tuple[UserSchema, str, str]:
        """Authenticate user and return user data with tokens"""
        users_collection = get_collection('users')
        
        # Find user by email
        user_doc = await users_collection.find_one({"email": login_data.email})
        if not user_doc:
            raise AuthenticationError("Invalid email or password")
        
        # Verify password
        if not verify_password(login_data.password, user_doc['password_hash']):
            raise AuthenticationError("Invalid email or password")
        
        # Check if user is active
        if not user_doc.get('is_active', True):
            raise AuthenticationError("User account is inactive")
        
        # Update last login
        await users_collection.update_one(
            {"_id": user_doc['_id']},
            {"$set": {"last_login": datetime.now(timezone.utc)}}
        )
        
        # Create tokens
        user_id = str(user_doc['_id'])
        access_token = create_access_token(identity=user_id)
        refresh_token = create_refresh_token(identity=user_id)
        
        # Convert user doc to schema
        user_dict = convert_objectids_to_strings(user_doc)
        user_schema = UserSchema(**user_dict)
        
        return user_schema, access_token, refresh_token
    
    @staticmethod
    async def get_user_by_id(user_id: str) -> Optional[UserSchema]:
        """Get user by ID"""
        users_collection = get_collection('users')
        
        user_doc = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user_doc:
            return None
        
        user_dict = convert_objectids_to_strings(user_doc)
        return UserSchema(**user_dict)
    
    @staticmethod
    async def refresh_token(user_id: str) -> str:
        """Generate new access token for user"""
        users_collection = get_collection('users')
        
        # Verify user still exists and is active
        user_doc = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user_doc or not user_doc.get('is_active', True):
            raise AuthenticationError("User account not found or inactive")
        
        return create_access_token(identity=user_id)
    
    @staticmethod
    async def update_user_role(user_id: str, new_role: UserRole, workspace_id: str) -> UserSchema:
        """Update user role (admin only)"""
        users_collection = get_collection('users')
        
        result = await users_collection.update_one(
            {"_id": ObjectId(user_id), "workspace_id": ObjectId(workspace_id)},
            {"$set": {"role": new_role.value, "updated_at": datetime.now(timezone.utc)}}
        )
        
        if result.matched_count == 0:
            raise ValidationError("User not found")
        
        updated_user = await users_collection.find_one({"_id": ObjectId(user_id)})
        user_dict = convert_objectids_to_strings(updated_user)
        return UserSchema(**user_dict)
    
    @staticmethod
    async def deactivate_user(user_id: str, workspace_id: str) -> UserSchema:
        """Deactivate a user (admin only)"""
        users_collection = get_collection('users')
        
        result = await users_collection.update_one(
            {"_id": ObjectId(user_id), "workspace_id": ObjectId(workspace_id)},
            {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}}
        )
        
        if result.matched_count == 0:
            raise ValidationError("User not found")
        
        updated_user = await users_collection.find_one({"_id": ObjectId(user_id)})
        user_dict = convert_objectids_to_strings(updated_user)
        return UserSchema(**user_dict)