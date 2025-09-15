from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from datetime import datetime, timezone
from bson import ObjectId
from app.core.database import async_route, get_collection
from app.schemas.base import (
    CreateAgentRequest, UpdateAgentRequest, AgentSchema, 
    PaginationRequest, PaginationResponse
)
from app.utils.validation import validate_json, validate_query_params, create_success_response, sanitize_update_data
from app.utils.auth import require_workspace_access
from app.utils.pagination import paginate_collection, convert_objectids_to_strings
from app.core.errors import NotFoundError, ValidationError

bp = Blueprint('agents', __name__)

@bp.route('', methods=['POST'])
@jwt_required()
@require_workspace_access
@async_route
@validate_json(CreateAgentRequest)
async def create_agent(validated_data: CreateAgentRequest, workspace_id: str, current_user_id: str):
    """Create a new agent"""
    agents_collection = get_collection('agents')
    
    agent_data = {
        'workspace_id': ObjectId(workspace_id),
        'created_by': ObjectId(current_user_id),
        'name': validated_data.name,
        'description': validated_data.description,
        'type': validated_data.type,
        'config': validated_data.config,
        'status': 'idle',
        'metrics': {},
        'created_at': datetime.now(timezone.utc),
        'updated_at': datetime.now(timezone.utc),
        'last_run': None
    }
    
    result = await agents_collection.insert_one(agent_data)
    
    # Fetch the created agent
    created_agent = await agents_collection.find_one({'_id': result.inserted_id})
    agent_dict = convert_objectids_to_strings(created_agent)
    
    return jsonify(create_success_response(
        data=AgentSchema(**agent_dict).dict(),
        message="Agent created successfully"
    )), 201

@bp.route('', methods=['GET'])
@jwt_required()
@require_workspace_access
@async_route
@validate_query_params(PaginationRequest)
async def list_agents(query_params: PaginationRequest, workspace_id: str, current_user_id: str):
    """List agents with pagination"""
    agents_collection = get_collection('agents')
    
    # Build filter query
    filter_query = {}
    if request.args.get('type'):
        filter_query['type'] = request.args.get('type')
    if request.args.get('status'):
        filter_query['status'] = request.args.get('status')
    
    documents, pagination_info = await paginate_collection(
        agents_collection, 
        workspace_id, 
        query_params, 
        filter_query
    )
    
    # Convert to schemas
    agents = [AgentSchema(**convert_objectids_to_strings(doc)).dict() for doc in documents]
    
    return jsonify(create_success_response({
        'agents': agents,
        'pagination': pagination_info.dict()
    })), 200

@bp.route('/<agent_id>', methods=['GET'])
@jwt_required()
@require_workspace_access
@async_route
async def get_agent(agent_id: str, workspace_id: str, current_user_id: str):
    """Get a specific agent"""
    if not ObjectId.is_valid(agent_id):
        raise ValidationError("Invalid agent ID format")
    
    agents_collection = get_collection('agents')
    
    agent = await agents_collection.find_one({
        '_id': ObjectId(agent_id),
        'workspace_id': ObjectId(workspace_id)
    })
    
    if not agent:
        raise NotFoundError("Agent not found")
    
    agent_dict = convert_objectids_to_strings(agent)
    return jsonify(create_success_response(
        data=AgentSchema(**agent_dict).dict()
    )), 200

@bp.route('/<agent_id>', methods=['PUT'])
@jwt_required()
@require_workspace_access
@async_route
@validate_json(UpdateAgentRequest)
async def update_agent(agent_id: str, validated_data: UpdateAgentRequest, workspace_id: str, current_user_id: str):
    """Update an agent"""
    if not ObjectId.is_valid(agent_id):
        raise ValidationError("Invalid agent ID format")
    
    agents_collection = get_collection('agents')
    
    # Check if agent exists
    existing_agent = await agents_collection.find_one({
        '_id': ObjectId(agent_id),
        'workspace_id': ObjectId(workspace_id)
    })
    
    if not existing_agent:
        raise NotFoundError("Agent not found")
    
    # Prepare update data
    update_data = sanitize_update_data(
        validated_data.dict(), 
        ['name', 'description', 'config', 'status']
    )
    update_data['updated_at'] = datetime.now(timezone.utc)
    
    # Update agent
    await agents_collection.update_one(
        {'_id': ObjectId(agent_id)},
        {'$set': update_data}
    )
    
    # Fetch updated agent
    updated_agent = await agents_collection.find_one({'_id': ObjectId(agent_id)})
    agent_dict = convert_objectids_to_strings(updated_agent)
    
    return jsonify(create_success_response(
        data=AgentSchema(**agent_dict).dict(),
        message="Agent updated successfully"
    )), 200

@bp.route('/<agent_id>', methods=['DELETE'])
@jwt_required()
@require_workspace_access
@async_route
async def delete_agent(agent_id: str, workspace_id: str, current_user_id: str):
    """Delete an agent"""
    if not ObjectId.is_valid(agent_id):
        raise ValidationError("Invalid agent ID format")
    
    agents_collection = get_collection('agents')
    
    result = await agents_collection.delete_one({
        '_id': ObjectId(agent_id),
        'workspace_id': ObjectId(workspace_id)
    })
    
    if result.deleted_count == 0:
        raise NotFoundError("Agent not found")
    
    return jsonify(create_success_response(
        message="Agent deleted successfully"
    )), 200

@bp.route('/<agent_id>/run', methods=['POST'])
@jwt_required()
@require_workspace_access
@async_route
async def run_agent(agent_id: str, workspace_id: str, current_user_id: str):
    """Run an agent (stub implementation)"""
    if not ObjectId.is_valid(agent_id):
        raise ValidationError("Invalid agent ID format")
    
    agents_collection = get_collection('agents')
    
    # Check if agent exists
    agent = await agents_collection.find_one({
        '_id': ObjectId(agent_id),
        'workspace_id': ObjectId(workspace_id)
    })
    
    if not agent:
        raise NotFoundError("Agent not found")
    
    # Update agent status and last_run
    await agents_collection.update_one(
        {'_id': ObjectId(agent_id)},
        {
            '$set': {
                'status': 'running',
                'last_run': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            }
        }
    )
    
    # TODO: Queue background job for actual agent execution
    # For now, this is just a stub that updates the status
    
    return jsonify(create_success_response(
        message="Agent run initiated successfully",
        data={'agent_id': agent_id, 'status': 'running'}
    )), 200

@bp.route('/<agent_id>/stop', methods=['POST'])
@jwt_required()
@require_workspace_access
@async_route
async def stop_agent(agent_id: str, workspace_id: str, current_user_id: str):
    """Stop a running agent (stub implementation)"""
    if not ObjectId.is_valid(agent_id):
        raise ValidationError("Invalid agent ID format")
    
    agents_collection = get_collection('agents')
    
    # Update agent status
    result = await agents_collection.update_one(
        {
            '_id': ObjectId(agent_id),
            'workspace_id': ObjectId(workspace_id)
        },
        {
            '$set': {
                'status': 'idle',
                'updated_at': datetime.now(timezone.utc)
            }
        }
    )
    
    if result.matched_count == 0:
        raise NotFoundError("Agent not found")
    
    return jsonify(create_success_response(
        message="Agent stopped successfully",
        data={'agent_id': agent_id, 'status': 'idle'}
    )), 200