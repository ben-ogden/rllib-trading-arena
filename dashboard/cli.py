"""
Dashboard CLI for RLlib Trading Demo

Simple CLI entry point for launching the trading dashboard.
"""

import click
import subprocess
import sys
import logging

logger = logging.getLogger(__name__)


@click.command()
@click.option('--port', '-p',
              type=int,
              default=8501,
              help='Port to run dashboard on')
@click.option('--host', '-h',
              default='localhost',
              help='Host to run dashboard on')
@click.option('--config', '-c',
              type=click.Path(exists=True),
              default='configs/trading_config.yaml',
              help='Path to configuration file')
def main(port: int, host: str, config: str) -> None:
    """Launch the RLlib Trading Demo Dashboard."""
    logger.info(f"Starting dashboard on {host}:{port}")
    
    try:
        # Launch Streamlit dashboard
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            'dashboard/trading_dashboard.py',
            '--server.port', str(port),
            '--server.address', host,
            '--server.headless', 'true'
        ]
        
        subprocess.run(cmd)
        
    except Exception as e:
        logger.error(f"Failed to launch dashboard: {e}")
        raise click.Abort()


if __name__ == '__main__':
    main()

