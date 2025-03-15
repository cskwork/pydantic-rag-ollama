"""Helper utility functions."""
import logging
import re
import unicodedata
from typing import Optional

import logfire

from config.settings import Settings, get_settings


def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """Slugify a string, to make it URL friendly.
    
    Args:
        value (str): String to slugify.
        separator (str): Separator character.
        unicode (bool, optional): Whether to preserve unicode. Defaults to False.
        
    Returns:
        str: Slugified string.
    """
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `žlutý` => `zluty`
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(rf'[{separator}\s]+', separator, value)


def setup_logging(settings: Optional[Settings] = None):
    """Configure logging for the application.
    
    Args:
        settings (Optional[Settings], optional): Settings override. Defaults to None.
    """
    if settings is None:
        settings = get_settings()
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    # Configure logfire if token is present
    if settings.logfire_token:
        logfire.configure(
            send_to_logfire='if-token-present', 
            token=settings.logfire_token
        )
        try:
            # This might not be available in older versions
            logfire.instrument_asyncpg()
        except AttributeError:
            logging.warning("logfire.instrument_asyncpg not available - skipping")
    else:
        logfire.configure(send_to_logfire='never')
