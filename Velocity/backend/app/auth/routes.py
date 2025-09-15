from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.core.database import async_route
from app.core.auth_service import AuthService
from app.schemas.base import CreateUserRequest, LoginRequest, LoginResponse, RefreshTokenRequest
from app.utils.validation import validate_json, create_success_response
from app.utils.auth import get_workspace_id, require_admin, get_current_user_id
from app.models.base import UserRole

bp = Blueprint('auth', __name__)

@bp.route('/register', methods=['POST'])
@async_route
@validate_json(CreateUserRequest)
async def register(validated_data: CreateUserRequest):
    """Register a new user"""
    workspace_id = get_workspace_id()
    
    user = await AuthService.create_user(validated_data, workspace_id)
    
    return jsonify(create_success_response(
        data=user.dict(),
        message="User created successfully"
    )), 201

@bp.route('/login', methods=['POST'])
@async_route
@validate_json(LoginRequest)
async def login(validated_data: LoginRequest):
    """Login a user"""
    user, access_token, refresh_token = await AuthService.authenticate_user(validated_data)
    
    response_data = LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user
    )
    
    return jsonify(create_success_response(data=response_data.dict())), 200

@bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
@async_route
def refresh():
    """Refresh access token"""
    current_user_id = get_jwt_identity()
    new_token = await AuthService.refresh_token(current_user_id)
    
    return jsonify(create_success_response(
        data={"access_token": new_token}
    )), 200

@bp.route('/me', methods=['GET'])
@jwt_required()
@async_route
async def get_current_user():
    """Get current user profile"""
    current_user_id = get_current_user_id()
    user = await AuthService.get_user_by_id(current_user_id)
    
    if not user:
        return jsonify({
            'ok': False,
            'error': {
                'code': 'USER_NOT_FOUND',
                'message': 'User not found'
            }
        }), 404
    
    return jsonify(create_success_response(data=user.dict())), 200

@bp.route('/users/<user_id>/role', methods=['PUT'])
@jwt_required()
@require_admin
@async_route
async def update_user_role(user_id: str, workspace_id: str, current_user_id: str):
    """Update user role (admin only)"""
    data = request.get_json()
    
    if 'role' not in data:
        return jsonify({
            'ok': False,
            'error': {
                'code': 'MISSING_ROLE',
                'message': 'Role is required'
            }
        }), 400
    
    try:
        new_role = UserRole(data['role'])
    except ValueError:
        return jsonify({
            'ok': False,
            'error': {
                'code': 'INVALID_ROLE',
                'message': f'Invalid role. Must be one of: {[role.value for role in UserRole]}'
            }
        }), 400
    
    user = await AuthService.update_user_role(user_id, new_role, workspace_id)
    
    return jsonify(create_success_response(
        data=user.dict(),
        message="User role updated successfully"
    )), 200

@bp.route('/users/<user_id>/deactivate', methods=['PUT'])
@jwt_required()
@require_admin
@async_route
async def deactivate_user(user_id: str, workspace_id: str, current_user_id: str):
    """Deactivate a user (admin only)"""
    user = await AuthService.deactivate_user(user_id, workspace_id)
    
    return jsonify(create_success_response(
        data=user.dict(),
        message="User deactivated successfully"
    )), 200

@bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout user (client-side token removal)"""
    # In a real implementation, you might want to blacklist the token
    return jsonify(create_success_response(
        message="Logged out successfully"
    )), 200