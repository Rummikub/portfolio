from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from datetime import datetime, timezone
from bson import ObjectId
from app.core.database import async_route, get_collection
from app.schemas.base import (
    CreateResearchRequest, ResearchSchema, 
    PaginationRequest, PaginationResponse
)
from app.utils.validation import validate_json, validate_query_params, create_success_response, sanitize_update_data
from app.utils.auth import require_workspace_access
from app.utils.pagination import paginate_collection, convert_objectids_to_strings
from app.core.errors import NotFoundError, ValidationError

bp = Blueprint('research', __name__)

@bp.route('', methods=['POST'])
@jwt_required()
@require_workspace_access
@async_route
@validate_json(CreateResearchRequest)
async def create_research(validated_data: CreateResearchRequest, workspace_id: str, current_user_id: str):
    """Create a new research entry"""
    research_collection = get_collection('research')
    
    research_data = {
        'workspace_id': ObjectId(workspace_id),
        'created_by': ObjectId(current_user_id),
        'title': validated_data.title,
        'type': validated_data.type,
        'data': validated_data.data,
        'source': validated_data.source,
        'confidence_score': validated_data.confidence_score,
        'tags': validated_data.tags,
        'created_at': datetime.now(timezone.utc),
        'updated_at': datetime.now(timezone.utc)
    }
    
    result = await research_collection.insert_one(research_data)
    
    created_research = await research_collection.find_one({'_id': result.inserted_id})
    research_dict = convert_objectids_to_strings(created_research)
    
    return jsonify(create_success_response(
        data=ResearchSchema(**research_dict).dict(),
        message="Research entry created successfully"
    )), 201

@bp.route('', methods=['GET'])
@jwt_required()
@require_workspace_access
@async_route
@validate_query_params(PaginationRequest)
async def list_research(query_params: PaginationRequest, workspace_id: str, current_user_id: str):
    """List research entries with pagination"""
    research_collection = get_collection('research')
    
    filter_query = {}
    if request.args.get('type'):
        filter_query['type'] = request.args.get('type')
    if request.args.get('source'):
        filter_query['source'] = request.args.get('source')
    if request.args.get('tag'):
        filter_query['tags'] = {'$in': [request.args.get('tag')]}
    
    documents, pagination_info = await paginate_collection(
        research_collection, 
        workspace_id, 
        query_params, 
        filter_query
    )
    
    research_items = [ResearchSchema(**convert_objectids_to_strings(doc)).dict() for doc in documents]
    
    return jsonify(create_success_response({
        'research': research_items,
        'pagination': pagination_info.dict()
    })), 200

@bp.route('/<research_id>', methods=['GET'])
@jwt_required()
@require_workspace_access
@async_route
async def get_research(research_id: str, workspace_id: str, current_user_id: str):
    """Get a specific research entry"""
    if not ObjectId.is_valid(research_id):
        raise ValidationError("Invalid research ID format")
    
    research_collection = get_collection('research')
    
    research = await research_collection.find_one({
        '_id': ObjectId(research_id),
        'workspace_id': ObjectId(workspace_id)
    })
    
    if not research:
        raise NotFoundError("Research entry not found")
    
    research_dict = convert_objectids_to_strings(research)
    return jsonify(create_success_response(
        data=ResearchSchema(**research_dict).dict()
    )), 200

@bp.route('/<research_id>', methods=['DELETE'])
@jwt_required()
@require_workspace_access
@async_route
async def delete_research(research_id: str, workspace_id: str, current_user_id: str):
    """Delete a research entry"""
    if not ObjectId.is_valid(research_id):
        raise ValidationError("Invalid research ID format")
    
    research_collection = get_collection('research')
    
    result = await research_collection.delete_one({
        '_id': ObjectId(research_id),
        'workspace_id': ObjectId(workspace_id)
    })
    
    if result.deleted_count == 0:
        raise NotFoundError("Research entry not found")
    
    return jsonify(create_success_response(
        message="Research entry deleted successfully"
    )), 200

@bp.route('/analytics', methods=['GET'])
@jwt_required()
@require_workspace_access
@async_route
async def get_research_analytics(workspace_id: str, current_user_id: str):
    """Get research analytics for the workspace"""
    research_collection = get_collection('research')
    
    # Aggregate analytics data
    pipeline = [
        {'$match': {'workspace_id': ObjectId(workspace_id)}},
        {
            '$group': {
                '_id': '$type',
                'count': {'$sum': 1},
                'avg_confidence': {'$avg': '$confidence_score'}
            }
        }
    ]
    
    analytics = await research_collection.aggregate(pipeline).to_list(length=None)
    
    # Get total count
    total_count = await research_collection.count_documents({'workspace_id': ObjectId(workspace_id)})
    
    return jsonify(create_success_response({
        'total_research_entries': total_count,
        'by_type': analytics
    })), 200