"""
Main script to run SyncFit-Storyblok synchronization
"""
import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from storyblok_integration.data_sync import data_sync_manager
from src.data_simulator import simulate_wearable_data

# Import train_churn_model with proper handling
import sys
import os
# Ensure src is in the path
if 'src' not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Now import the function directly
import src.feature_builder as fb
import src.churn_model as cm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('syncfit_storyblok.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup necessary directories and files"""
    directories = ['data', 'models', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    logger.info("Environment setup completed")

def generate_sample_data():
    """Generate sample wearable data if it doesn't exist"""
    data_file = Path("data/synthetic_wearable_logs.csv")
    if not data_file.exists():
        logger.info("Generating sample wearable data...")
        simulate_wearable_data(n_users=100, n_days=30)
        logger.info("Sample data generated successfully")
    else:
        logger.info("Sample data already exists")

def train_model_if_needed():
    """Train churn model if it doesn't exist"""
    model_file = Path("models/syncfit_churn_model.pkl")
    if not model_file.exists():
        logger.info("Training churn prediction model...")
        # Call the train function directly from the module
        cm.train_churn_model()
        logger.info("Model trained successfully")
    else:
        logger.info("Churn model already exists")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='SyncFit-Storyblok Integration')
    parser.add_argument(
        '--action',
        choices=['setup', 'sync', 'sync-users', 'sync-metrics', 'sync-alerts', 
                'sync-predictions', 'status', 'generate-data', 'train-model', 'full'],
        default='full',
        help='Action to perform'
    )
    parser.add_argument(
        '--csv-path',
        default='data/synthetic_wearable_logs.csv',
        help='Path to CSV file with wearable data'
    )
    parser.add_argument(
        '--api-server',
        action='store_true',
        help='Start the API server after sync'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting SyncFit-Storyblok Integration - Action: {args.action}")
        
        if args.action == 'setup':
            setup_environment()
            generate_sample_data()
            train_model_if_needed()
            logger.info("Setup completed successfully")
            
        elif args.action == 'generate-data':
            generate_sample_data()
            
        elif args.action == 'train-model':
            train_model_if_needed()
            
        elif args.action == 'sync':
            logger.info("Starting full data synchronization...")
            results = data_sync_manager.full_sync(args.csv_path)
            logger.info(f"Synchronization completed: {results}")
            
        elif args.action == 'sync-users':
            logger.info("Syncing user profiles...")
            results = data_sync_manager.create_user_profiles_from_data(args.csv_path)
            logger.info(f"Created {len(results)} user profiles")
            
        elif args.action == 'sync-metrics':
            logger.info("Syncing health metrics...")
            results = data_sync_manager.sync_wearable_data_to_storyblok(args.csv_path)
            logger.info(f"Synced {len(results)} health metrics")
            
        elif args.action == 'sync-alerts':
            logger.info("Syncing guardian alerts...")
            results = data_sync_manager.sync_guardian_alerts(args.csv_path)
            logger.info(f"Created {len(results)} alerts")
            
        elif args.action == 'sync-predictions':
            logger.info("Syncing churn predictions...")
            results = data_sync_manager.sync_churn_predictions(args.csv_path)
            logger.info(f"Created {len(results)} predictions")
            
        elif args.action == 'status':
            status = data_sync_manager.get_sync_status()
            logger.info("Current synchronization status:")
            for key, value in status.items():
                logger.info(f"  {key}: {value}")
                
        elif args.action == 'full':
            # Complete setup and sync
            logger.info("Running full setup and synchronization...")
            setup_environment()
            generate_sample_data()
            train_model_if_needed()
            
            logger.info("Starting data synchronization...")
            results = data_sync_manager.full_sync(args.csv_path)
            
            logger.info("Synchronization completed successfully!")
            logger.info("Summary:")
            logger.info(f"  User profiles: {len(results.get('user_profiles', []))}")
            logger.info(f"  Health metrics: {len(results.get('health_metrics', []))}")
            logger.info(f"  Alerts: {len(results.get('alerts', []))}")
            logger.info(f"  Predictions: {len(results.get('predictions', []))}")
            
            # Get final status
            status = data_sync_manager.get_sync_status()
            logger.info("\nFinal Status:")
            for key, value in status.items():
                logger.info(f"  {key}: {value}")
        
        # Start API server if requested
        if args.api_server:
            logger.info("Starting API server...")
            import uvicorn
            from storyblok_integration.api_server import app
            uvicorn.run(app, host="0.0.0.0", port=8000)
            
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()
