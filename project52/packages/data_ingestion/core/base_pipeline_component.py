from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

class BasePipelineComponent(ABC):
    """Abstract base class for all pipeline components in the data ingestion platform.
    
    This class defines the interface that all pipeline components must implement.
    It provides a standard way to process data through the pipeline, with support for
    configuration, logging, and error handling.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the pipeline component with a configuration dictionary.
        
        Args:
            config: Dictionary containing component-specific settings.
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.logger = logging.getLogger(self.name)
        self.validate_config()

    def validate_config(self) -> None:
        """Validate the component's configuration.
        
        This method can be overridden by subclasses to check for required configuration
        settings. By default, it does nothing.
        """
        pass

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process the input data and return the result.
        
        This method must be implemented by all subclasses to define the specific
        data processing logic for the component.
        
        Args:
            data: The input data to be processed.
        
        Returns:
            The processed data.
        
        Raises:
            Exception: If an error occurs during processing.
        """
        pass

    def __str__(self) -> str:
        """Return a string representation of the component.
        
        Returns:
            The name of the component.
        """
        return self.name