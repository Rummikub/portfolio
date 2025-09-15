from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from datetime import datetime, timezone
from bson import ObjectId
from app.core.database import async_route, get_collection
from app.schemas.base import (
    CreateDesignPartnerRequest, DesignPartnerSchema, 
    PaginationRequest, PaginationResponse
)
from app.utils.validation import validate_json, validate_query_params, create_success_response, sanitize_update_data
from app.utils.auth import require_workspace_access
from app.utils.pagination import paginate_collection, convert_objectids_to_strings
from app.core.errors import NotFoundError, ValidationError

bp = Blueprint('design_partners', __name__)

@bp.route('', methods=['POST'])
@jwt_required()
@require_workspace_access
@async_route
@validate_json(CreateDesignPartnerRequest)
async def create_design_partner(validated_data: CreateDesignPartnerRequest, workspace_id: str, current_user_id: str):
    """Create a new design partner"""
    partners_collection = get_collection('design_partners')
    
    partner_data = {
        'workspace_id': ObjectId(workspace_id),
        'created_by': ObjectId(current_user_id),
        'name': validated_data.name,
        'email': validated_data.email,
        'company': validated_data.company,
        'industry': validated_data.industry,
        'status': validated_data.status,
        'contact_history': [],
        'notes': validated_data.notes,
        'created_at': datetime.now(timezone.utc),
        'updated_at': datetime.now(timezone.utc)
    }
    
    result = await partners_collection.insert_one(partner_data)
    
    created_partner = await partners_collection.find_one({'_id': result.inserted_id})
    partner_dict = convert_objectids_to_strings(created_partner)
    
    return jsonify(create_success_response(
        data=DesignPartnerSchema(**partner_dict).dict(),
        message="Design partner created successfully"
    )), 201

@bp.route('', methods=['GET'])
@jwt_required()
@require_workspace_access
@async_route
@validate_query_params(PaginationRequest)
async def list_design_partners(query_params: PaginationRequest, workspace_id: str, current_user_id: str):
    """List design partners with pagination"""
    partners_collection = get_collection('design_partners')
    
    filter_query = {}
    if request.args.get('status'):
        filter_query['status'] = request.args.get('status')
    if request.args.get('industry'):
        filter_query['industry'] = request.args.get('industry')
    
    documents, pagination_info = await paginate_collection(
        partners_collection, 
        workspace_id, 
        query_params, 
        filter_query
    )
    
    partners = [DesignPartnerSchema(**convert_objectids_to_strings(doc)).dict() for doc in documents]
    
    return jsonify(create_success_response({
        'design_partners': partners,
        'pagination': pagination_info.dict()
    })), 200

@bp.route('/<partner_id>', methods=['GET'])
@jwt_required()
@require_workspace_access
@async_route
async def get_design_partner(partner_id: str, workspace_id: str, current_user_id: str):
    """Get a specific design partner"""
    if not ObjectId.is_valid(partner_id):
        raise ValidationError("Invalid partner ID format")
    
    partners_collection = get_collection('design_partners')
    
    partner = await partners_collection.find_one({
        '_id': ObjectId(partner_id),
        'workspace_id': ObjectId(workspace_id)
    })
    
    if not partner:
        raise NotFoundError("Design partner not found")
    
    partner_dict = convert_objectids_to_strings(partner)
    return jsonify(create_success_response(
        data=DesignPartnerSchema(**partner_dict).dict()
    )), 200

@bp.route('/<partner_id>/contact', methods=['POST'])
@jwt_required()
@require_workspace_access
@async_route
async def add_contact_log(partner_id: str, workspace_id: str, current_user_id: str):
    """Add a contact log entry to a design partner"""
    if not ObjectId.is_valid(partner_id):
        raise ValidationError("Invalid partner ID format")
    
    data = request.get_json()
    if not data or 'type' not in data or 'notes' not in data:
        raise ValidationError("Contact type and notes are required")
    
    partners_collection = get_collection('design_partners')
    
    contact_entry = {
        'type': data['type'],
        'notes': data['notes'],
        'date': datetime.now(timezone.utc),
        'contacted_by': ObjectId(current_user_id)
    }
    
    result = await partners_collection.update_one(
        {
            '_id': ObjectId(partner_id),
            'workspace_id': ObjectId(workspace_id)
        },
        {
            '$push': {'contact_history': contact_entry},
            '$set': {'updated_at': datetime.now(timezone.utc)}
        }
    )
    
    if result.matched_count == 0:
        raise NotFoundError("Design partner not found")
    
    return jsonify(create_success_response(
        message="Contact log added successfully"
    )), 201