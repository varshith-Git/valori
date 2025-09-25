"""
Custom exceptions for the Valori vector database.
"""


class ValoriError(Exception):
    """Base exception class for all Valori-related errors."""
    pass


class StorageError(ValoriError):
    """Exception raised for storage-related errors."""
    pass


class ValoriIndexError(ValoriError):
    """Exception raised for index-related errors."""
    pass


class QuantizationError(ValoriError):
    """Exception raised for quantization-related errors."""
    pass


class PersistenceError(ValoriError):
    """Exception raised for persistence-related errors."""
    pass


class ValidationError(ValoriError):
    """Exception raised for validation errors."""
    pass


class ConfigurationError(ValoriError):
    """Exception raised for configuration errors."""
    pass


class ResourceError(ValoriError):
    """Exception raised for resource-related errors."""
    pass


class ParsingError(ValoriError):
    """Exception raised for document parsing errors."""
    pass


class ProcessingError(ValoriError):
    """Exception raised for document processing errors."""
    pass
