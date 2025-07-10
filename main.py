import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from video_processor import VideoProcessor
from database_manager import DatabaseManager
from logger_manager import LoggerManager
import utils


def setup_directories():
    directories = [
        'logs',
        'logs/entries',
        'logs/exits',
        'data',
        'data/faces'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Intelligent Face Tracker')
    parser.add_argument('--video', '-v', 
                       default='video_sample1.mp4',
                       help='Path to video file or RTSP stream URL')
    parser.add_argument('--config', '-c',
                       default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--debug', '-d',
                       action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    setup_directories()
    config = utils.load_config(args.config)
    if not config:
        print(f"Error: Could not load configuration from {args.config}")
        return 1
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger_manager = LoggerManager(log_level=log_level)
    logger = logger_manager.get_logger('main')
    
    logger.info("Starting Intelligent Face Tracker")
    logger.info(f"Video source: {args.video}")
    logger.info(f"Configuration: {args.config}")
    
    try:
        db_manager = DatabaseManager(config['database']['path'])
        db_manager.initialize_database()
        logger.info("Database initialized successfully")
        processor = VideoProcessor(
            video_source=args.video,
            config=config,
            db_manager=db_manager,
            logger_manager=logger_manager
        )
        processor.process_video()
        stats = db_manager.get_visitor_statistics()
        logger.info("Processing completed successfully")
        logger.info(f"Total unique visitors: {stats['unique_visitors']}")
        logger.info(f"Total entries: {stats['total_entries']}")
        logger.info(f"Total exits: {stats['total_exits']}")
        
        print("\n" + "="*50)
        print("FACE TRACKING SUMMARY")
        print("="*50)
        print(f"Unique Visitors: {stats['unique_visitors']}")
        print(f"Total Entries: {stats['total_entries']}")
        print(f"Total Exits: {stats['total_exits']}")
        print(f"Currently Inside: {stats['total_entries'] - stats['total_exits']}")
        print("="*50)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
