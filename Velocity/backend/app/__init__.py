from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import os
from datetime import timedelta

def create_app(config_name: str = None) -> Flask:
    app = Flask(__name__)
    
    # Load configuration
    config_name = config_name or os.getenv('FLASK_ENV', 'development')
    from app.core.config import config
    app.config.from_object(config.get(config_name, config['default']))
    
    # Initialize extensions
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Initialize JWT
    jwt = JWTManager(app)
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(seconds=app.config['JWT_ACCESS_TOKEN_EXPIRES'])
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(seconds=app.config['JWT_REFRESH_TOKEN_EXPIRES'])
    
    # JWT error handlers
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({
            'ok': False,
            'error': {
                'code': 'TOKEN_EXPIRED',
                'message': 'The token has expired'
            }
        }), 401

    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({
            'ok': False,
            'error': {
                'code': 'INVALID_TOKEN',
                'message': 'Invalid token'
            }
        }), 401

    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({
            'ok': False,
            'error': {
                'code': 'MISSING_TOKEN',
                'message': 'Authorization token is required'
            }
        }), 401
    
    # Initialize MongoDB connection
    from app.core.database import init_db
    init_db(app)
    
    # Register blueprints
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    
    from app.agents import bp as agents_bp
    app.register_blueprint(agents_bp, url_prefix='/api/agents')
    
    from app.docs import bp as docs_bp
    app.register_blueprint(docs_bp, url_prefix='/api/docs')
    
    from app.design_partners import bp as design_partners_bp
    app.register_blueprint(design_partners_bp, url_prefix='/api/design-partners')
    
    from app.research import bp as research_bp
    app.register_blueprint(research_bp, url_prefix='/api/research')
    
    # Global error handlers
    from app.core.errors import register_error_handlers
    register_error_handlers(app)
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return jsonify({
            'ok': True,
            'status': 'healthy',
            'service': 'agentic-system-backend'
        })
    
    return app