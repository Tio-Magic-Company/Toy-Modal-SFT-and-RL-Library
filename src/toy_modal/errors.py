"""Project-specific exceptions normalized across transports."""


class ToyModalError(Exception):
    """Base class for all toy_modal exceptions."""


TinkerError = ToyModalError


class APIError(ToyModalError):
    """Base class for API-related errors."""

    def __init__(self, message: str = "", *, body=None) -> None:
        super().__init__(message)
        self.body = body


class APIResponseValidationError(APIError):
    """Raised when a backend response does not match the expected schema."""


class APIStatusError(APIError):
    """Raised when a backend response maps to a non-2xx status."""


class APIConnectionError(APIError):
    """Raised when a backend cannot be reached."""


class APITimeoutError(APIConnectionError):
    """Raised when waiting for a backend request times out."""


class BackendUnavailableError(ToyModalError):
    """The configured backend cannot be reached."""


class BadRequestError(APIStatusError):
    """The request payload or state transition is invalid."""


class AuthenticationError(APIStatusError):
    """Authentication credentials are missing or invalid."""


class PermissionDeniedError(APIStatusError):
    """The caller does not have permission to access the resource."""


class ConflictError(APIStatusError):
    """The requested operation conflicts with existing state."""


class NotFoundError(APIStatusError):
    """A requested run, job, or artifact was not found."""


class RunNotFoundError(NotFoundError):
    """A requested training run was not found."""


class CheckpointNotFoundError(NotFoundError):
    """A requested checkpoint was not found."""


class StaleModelSequenceError(ConflictError):
    """A training request targeted an old model sequence."""


class DependencyFailedError(ToyModalError):
    """A job dependency failed before this job could run."""


class TransientModalError(BackendUnavailableError):
    """A retryable Modal platform error occurred."""


class UnprocessableEntityError(APIStatusError):
    """The request is syntactically valid but semantically invalid."""


class RateLimitError(APIStatusError):
    """The backend rate limit has been exceeded."""


class InternalServerError(APIStatusError):
    """The backend returned an internal error."""


class SidecarError(ToyModalError):
    """Base class for sampling sidecar errors."""


class SidecarStartupError(SidecarError):
    """The sampling sidecar failed to start."""


class SidecarDiedError(SidecarError):
    """The sampling sidecar exited while requests were pending."""


class SidecarIPCError(SidecarError):
    """The client could not communicate with the sampling sidecar."""


class RequestFailedError(ToyModalError):
    """An asynchronous request completed in a failed state."""
