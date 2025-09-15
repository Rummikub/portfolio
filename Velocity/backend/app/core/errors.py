from flask import Flask, jsonify, request
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException
import traceback
from typing import Dict, Any

class APIError(Exception):
    """Base API Error class"""
    def __init__(self, message: str, code: str = "API_ERROR", status_code: int = 400, details: Dict[str, Any] = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ValidationError(APIError):
    """Validation error"""
    def __init__(self, message: str = "Validation failed", details: Dict[str, Any] = None):
        super().__init__(message, "VALIDATION_ERROR", 400, details)

class AuthenticationError(APIError):
    """Authentication error"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTHENTICATION_ERROR", 401)

class AuthorizationError(APIError):
    """Authorization error"""
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, "AUTHORIZATION_ERROR", 403)

class NotFoundError(APIError):
    """Resource not found error"""
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, "NOT_FOUND", 404)

class WorkspaceError(APIError):
    """Workspace-related error"""
    def __init__(self, message: str = "Workspace access denied"):
        super().__init__(message, "WORKSPACE_ERROR", 403)

def register_error_handlers(app: Flask):
    """Register global error handlers"""
    
    @app.errorhandler(APIError)
    def handle_api_error(error: APIError):
        return jsonify({
            'ok': False,
            'error': {
                'code': error.code,
                'message': error.message,
                'details': error.details
            }
        }), error.status_code
    
    @app.errorhandler(ValidationError)
    def handle_pydantic_validation_error(error: ValidationError):
        errors = []
        for err in error.errors():
            errors.append({
                'field': '.'.join(str(x) for x in err['loc']),
                'message': err['msg'],
                'type': err['type']
            })
        
        return jsonify({
            'ok': False,
            'error': {
                'code': 'VALIDATION_ERROR',
                'message': 'Validation failed',
                'details': errors
            }
        }), 400
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException):
        return jsonify({
            'ok': False,
            'error': {
                'code': 'HTTP_ERROR',
                'message': error.description,
                'details': {'status_code': error.code}
            }
        }), error.code
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        app.logger.error(f"Unexpected error: {error}")
        app.logger.error(traceback.format_exc())
        
        # Don't expose internal error details in production
        if app.config.get('DEBUG'):
            return jsonify({
                'ok': False,
                'error': {
                    'code': 'INTERNAL_ERROR',
                    'message': str(error),
                    'details': {'traceback': traceback.format_exc()}
                }
            }), 500
        else:
            return jsonify({
                'ok': False,
                'error': {
                    'code': 'INTERNAL_ERROR',
                    'message': 'An internal server error occurred'
                }
            }), 500

def create_error_response(code: str, message: str, status_code: int = 400, details: Dict[str, Any] = None):
    """Helper function to create consistent error responses"""
    return jsonify({
        'ok': False,
        'error': {
            'code': code,
            'message': message,
            'details': details or {}
        }
    }), status_code