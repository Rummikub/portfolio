import pytest
import json

@pytest.mark.asyncio
async def test_create_agent(client, auth_headers, sample_agent_data):
    """Test creating an agent"""
    response = client.post('/api/agents', 
                          json=sample_agent_data,
                          headers=auth_headers)
    
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data['ok'] is True
    assert 'data' in data
    assert data['data']['name'] == sample_agent_data['name']
    assert data['data']['type'] == sample_agent_data['type']
    assert data['data']['status'] == 'idle'

@pytest.mark.asyncio
async def test_list_agents(client, auth_headers, sample_agent_data):
    """Test listing agents"""
    # First create an agent
    client.post('/api/agents',
                json=sample_agent_data,
                headers=auth_headers)
    
    # Then list agents
    response = client.get('/api/agents', headers=auth_headers)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['ok'] is True
    assert 'data' in data
    assert 'agents' in data['data']
    assert len(data['data']['agents']) >= 1

@pytest.mark.asyncio
async def test_get_agent(client, auth_headers, sample_agent_data):
    """Test getting a specific agent"""
    # First create an agent
    create_response = client.post('/api/agents',
                                 json=sample_agent_data,
                                 headers=auth_headers)
    
    created_agent = json.loads(create_response.data)['data']
    agent_id = created_agent['id']
    
    # Then get the agent
    response = client.get(f'/api/agents/{agent_id}', headers=auth_headers)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['ok'] is True
    assert data['data']['id'] == agent_id
    assert data['data']['name'] == sample_agent_data['name']

@pytest.mark.asyncio
async def test_update_agent(client, auth_headers, sample_agent_data):
    """Test updating an agent"""
    # First create an agent
    create_response = client.post('/api/agents',
                                 json=sample_agent_data,
                                 headers=auth_headers)
    
    created_agent = json.loads(create_response.data)['data']
    agent_id = created_agent['id']
    
    # Update the agent
    update_data = {
        "name": "Updated Agent Name",
        "description": "Updated description"
    }
    
    response = client.put(f'/api/agents/{agent_id}',
                         json=update_data,
                         headers=auth_headers)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['ok'] is True
    assert data['data']['name'] == update_data['name']
    assert data['data']['description'] == update_data['description']

@pytest.mark.asyncio
async def test_run_agent(client, auth_headers, sample_agent_data):
    """Test running an agent"""
    # First create an agent
    create_response = client.post('/api/agents',
                                 json=sample_agent_data,
                                 headers=auth_headers)
    
    created_agent = json.loads(create_response.data)['data']
    agent_id = created_agent['id']
    
    # Run the agent
    response = client.post(f'/api/agents/{agent_id}/run',
                          headers=auth_headers)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['ok'] is True
    assert data['data']['status'] == 'running'

@pytest.mark.asyncio
async def test_delete_agent(client, auth_headers, sample_agent_data):
    """Test deleting an agent"""
    # First create an agent
    create_response = client.post('/api/agents',
                                 json=sample_agent_data,
                                 headers=auth_headers)
    
    created_agent = json.loads(create_response.data)['data']
    agent_id = created_agent['id']
    
    # Delete the agent
    response = client.delete(f'/api/agents/{agent_id}',
                           headers=auth_headers)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['ok'] is True
    
    # Verify agent is deleted
    get_response = client.get(f'/api/agents/{agent_id}',
                             headers=auth_headers)
    assert get_response.status_code == 404