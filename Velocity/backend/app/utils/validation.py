from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, ValidationError
from flask import request, jsonify
from functools import wraps
from app.core.errors import ValidationError as APIValidationError

def validate_json(schema: Type[BaseModel]):
    """Decorator to validate JSON request body against a Pydantic schema"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                if not request.is_json:
                    raise APIValidationError("Content-Type must be application/json")
                
                json_data = request.get_json()
                if json_data is None:
                    raise APIValidationError("Request body must be valid JSON")
                
                # Validate with Pydantic schema
                validated_data = schema(**json_data)
                kwargs['validated_data'] = validated_data
                
                return f(*args, **kwargs)
                
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    errors.append({
                        'field': '.'.join(str(x) for x in error['loc']),
                        'message': error['msg'],
                        'type': error['type']
                    })
                
                raise APIValidationError("Validation failed", details={'errors': errors})
                
        return wrapper
    return decorator

def validate_query_params(schema: Type[BaseModel]):
    """Decorator to validate query parameters against a Pydantic schema"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Convert query args to dict
                query_data = {}
                for key, value in request.args.items():
                    # Try to convert numeric strings to int
                    if value.isdigit():
                        query_data[key] = int(value)
                    elif value.lower() in ('true', 'false'):
                        query_data[key] = value.lower() == 'true'
                    else:
                        query_data[key] = value
                
                # Validate with Pydantic schema
                validated_data = schema(**query_data)
                kwargs['query_params'] = validated_data
                
                return f(*args, **kwargs)
                
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    errors.append({
                        'field': '.'.join(str(x) for x in error['loc']),
                        'message': error['msg'],
                        'type': error['type']
                    })
                
                raise APIValidationError("Query parameter validation failed", details={'errors': errors})
                
        return wrapper
    return decorator

def sanitize_update_data(data: Dict[str, Any], allowed_fields: list) -> Dict[str, Any]:
    """Sanitize update data to only include allowed fields and remove None values"""
    sanitized = {}
    for field in allowed_fields:
        if field in data and data[field] is not None:
            sanitized[field] = data[field]
    return sanitized

def create_success_response(data: Any = None, message: str = None) -> Dict[str, Any]:
    """Create a consistent success response"""
    response = {"ok": True}
    if data is not None:
        response["data"] = data
    if message:
        response["message"] = message
    return response