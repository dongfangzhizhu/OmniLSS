"""OmniLSS Pro boundary prototype.

This package intentionally does not import or depend on the GPL OmniLSS core
package. It talks to OmniLSS Core through the public gRPC API only.
"""

from .client import OmniLSSCoreClient

__all__ = ["OmniLSSCoreClient"]
