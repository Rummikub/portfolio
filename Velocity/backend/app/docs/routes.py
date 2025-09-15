from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from datetime import datetime, timezone
from bson import ObjectId
from app.core.database import async_route, get_collection
from app.schemas.base import (
    CreateDocRequest, DocSchema, 
    PaginationRequest, PaginationResponse
)
from app.utils.validation import validate_json, validate_query_params, create_success_response, sanitize_update_data
from app.utils.auth import require_workspace_access
from app.utils.pagination import paginate_collection, convert_objectids_to_strings
from app.core.errors import NotFoundError, ValidationError

bp = Blueprint('docs', __name__)

@bp.route('', methods=['POST'])
@jwt_required()
@require_workspace_access
@async_route
@validate_json(CreateDocRequest)
async def create_doc(validated_data: CreateDocRequest, workspace_id: str, current_user_id: str):
    """Create a new document"""
    docs_collection = get_collection('docs')
    
    doc_data = {
        'workspace_id': ObjectId(workspace_id),
        'created_by': ObjectId(current_user_id),
        'title': validated_data.title,
        'content': validated_data.content,
        'type': validated_data.type,
        'tags': validated_data.tags,
        'metadata': validated_data.metadata,
        'file_path': None,
        'created_at': datetime.now(timezone.utc),
        'updated_at': datetime.now(timezone.utc)
    }
    
    result = await docs_collection.insert_one(doc_data)
    
    created_doc = await docs_collection.find_one({'_id': result.inserted_id})
    doc_dict = convert_objectids_to_strings(created_doc)
    
    return jsonify(create_success_response(
        data=DocSchema(**doc_dict).dict(),
        message="Document created successfully"
    )), 201

@bp.route('', methods=['GET'])
@jwt_required()
@require_workspace_access
@async_route
@validate_query_params(PaginationRequest)
async def list_docs(query_params: PaginationRequest, workspace_id: str, current_user_id: str):
    """List documents with pagination"""
    docs_collection = get_collection('docs')
    
    filter_query = {}
    if request.args.get('type'):
        filter_query['type'] = request.args.get('type')
    if request.args.get('tag'):
        filter_query['tags'] = {'$in': [request.args.get('tag')]}
    
    documents, pagination_info = await paginate_collection(
        docs_collection, 
        workspace_id, 
        query_params, 
        filter_query
    )
    
    docs = [DocSchema(**convert_objectids_to_strings(doc)).dict() for doc in documents]
    
    return jsonify(create_success_response({
        'docs': docs,
        'pagination': pagination_info.dict()
    })), 200

@bp.route('/<doc_id>', methods=['GET'])
@jwt_required()
@require_workspace_access
@async_route
async def get_doc(doc_id: str, workspace_id: str, current_user_id: str):
    """Get a specific document"""
    if not ObjectId.is_valid(doc_id):
        raise ValidationError("Invalid document ID format")
    
    docs_collection = get_collection('docs')
    
    doc = await docs_collection.find_one({
        '_id': ObjectId(doc_id),
        'workspace_id': ObjectId(workspace_id)
    })
    
    if not doc:
        raise NotFoundError("Document not found")
    
    doc_dict = convert_objectids_to_strings(doc)
    return jsonify(create_success_response(
        data=DocSchema(**doc_dict).dict()
    )), 200

@bp.route('/<doc_id>', methods=['DELETE'])
@jwt_required()
@require_workspace_access
@async_route
async def delete_doc(doc_id: str, workspace_id: str, current_user_id: str):
    """Delete a document"""
    if not ObjectId.is_valid(doc_id):
        raise ValidationError("Invalid document ID format")
    
    docs_collection = get_collection('docs')
    
    result = await docs_collection.delete_one({
        '_id': ObjectId(doc_id),
        'workspace_id': ObjectId(workspace_id)
    })
    
    if result.deleted_count == 0:
        raise NotFoundError("Document not found")
    
    return jsonify(create_success_response(
        message="Document deleted successfully"
    )), 200