"""Image providers and the factory that selects one from settings.

To add a provider: implement `ImageGenerator`, add an `ImageProviderName` member, and
register its import path below. Pipeline code does not change.
"""

from __future__ import annotations

import importlib

from shortforge.config import ConfigError, ImageProviderName, Settings, get_settings
from shortforge.images.base import ImageGenerator, fit_to_frame

PROVIDERS: dict[ImageProviderName, str] = {
    ImageProviderName.CLOUDFLARE: "shortforge.images.cloudflare:CloudflareImages",
    ImageProviderName.REPLICATE: "shortforge.images.replicate:ReplicateImages",
}


def create_image_generator(settings: Settings | None = None) -> ImageGenerator:
    """Instantiate the image provider named by ``IMAGE_PROVIDER``.

    Raises:
        ConfigError: If the provider is unregistered or cannot be loaded.
    """
    settings = settings or get_settings()
    target = PROVIDERS.get(settings.image_provider)
    if target is None:
        raise ConfigError(f"No image provider registered for {settings.image_provider!r}")
    module_name, class_name = target.split(":")
    try:
        provider_cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(
            f"Image provider {settings.image_provider.value!r} failed to load: {exc}"
        ) from exc
    return provider_cls.from_settings(settings)


__all__ = ["PROVIDERS", "ImageGenerator", "create_image_generator", "fit_to_frame"]
