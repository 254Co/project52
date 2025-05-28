from abc import ABC, abstractmethod
from typing import Any, Dict
from core.base_pipeline_component import BasePipelineComponent
import logging

class BaseFetcher(BasePipelineComponent, ABC):
    """Abstract base class for all fetchers in the data ingestion platform.
    
    Fetchers are responsible for retrieving data from various sources, such as databases, APIs, or files.
    This class defines the interface that all fetchers must implement.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the fetcher with a configuration dictionary.
        
        Args:
            config: Dictionary containing fetcher-specific settings.
        """
        super().__init__(config)
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def fetch(self) -> Any:
        """Fetch data from the source.
        
        This method must be implemented by all subclasses to define how data is retrieved from the source.
        
        Returns:
            The data retrieved from the source in a format suitable for further processing.
        
        Raises:
            Exception: If an error occurs during data fetching.
        """
        pass

    def process(self, data: Any = None) -> Any:
        """Process method to integrate with the pipeline.
        
        Since fetchers typically start the pipeline, this method calls fetch() and returns the data.
        
        Args:
            data: Not used in fetchers, as they initiate the data flow.
        
        Returns:
            The data fetched from the source.
        """
        return self.fetch()