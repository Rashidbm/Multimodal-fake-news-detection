"""
Image Downloader for MultiGuard
===============================

Downloads and validates images from news articles.
Includes deduplication using perceptual hashing.
"""

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from urllib.parse import urlparse

import requests
from PIL import Image
from loguru import logger

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logger.warning("imagehash not installed. Perceptual hashing disabled.")


@dataclass
class DownloadedImage:
    """Represents a downloaded and validated image."""

    # File info
    local_path: str
    filename: str

    # Original source
    original_url: str
    source_domain: str

    # Image properties
    width: int
    height: int
    format: str
    file_size_bytes: int

    # Hashes for deduplication
    md5_hash: str
    perceptual_hash: Optional[str] = None

    # Validation
    is_valid: bool = True
    validation_errors: List[str] = None

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "local_path": self.local_path,
            "filename": self.filename,
            "original_url": self.original_url,
            "source_domain": self.source_domain,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "file_size_bytes": self.file_size_bytes,
            "md5_hash": self.md5_hash,
            "perceptual_hash": self.perceptual_hash,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }


class ImageDownloader:
    """
    Downloads and validates images from URLs.

    Features:
    - Downloads images with proper error handling
    - Validates image dimensions and format
    - Computes MD5 and perceptual hashes for deduplication
    - Rate limiting per domain
    - Saves images in organized directory structure
    """

    # Supported image formats
    SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "GIF"}

    # Common placeholder/logo patterns to filter out
    PLACEHOLDER_PATTERNS = [
        "placeholder",
        "default",
        "no-image",
        "noimage",
        "logo",
        "icon",
        "avatar",
        "thumbnail",
        "blank",
        "missing",
    ]

    def __init__(
        self,
        output_dir: str = "data/raw/images",
        min_width: int = 400,
        min_height: int = 300,
        max_file_size_mb: float = 10.0,
        rate_limit_seconds: float = 1.0,
        timeout: int = 30,
        user_agent: str = "MultiGuardBot/1.0 (academic research)"
    ):
        """
        Initialize the image downloader.

        Args:
            output_dir: Directory to save downloaded images
            min_width: Minimum image width in pixels
            min_height: Minimum image height in pixels
            max_file_size_mb: Maximum file size in megabytes
            rate_limit_seconds: Seconds between requests to same domain
            timeout: Request timeout in seconds
            user_agent: User agent for HTTP requests
        """
        self.output_dir = Path(output_dir)
        self.min_width = min_width
        self.min_height = min_height
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout = timeout
        self.user_agent = user_agent

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track seen hashes for deduplication
        self._seen_hashes: set = set()
        self._seen_perceptual_hashes: set = set()

        # Rate limiting
        self._last_request: Dict[str, float] = {}

        logger.info(f"Initialized ImageDownloader. Output: {self.output_dir}")

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")

    def _wait_for_rate_limit(self, domain: str) -> None:
        """Wait if necessary to respect rate limit."""
        now = time.time()
        last_request = self._last_request.get(domain, 0)
        elapsed = now - last_request

        if elapsed < self.rate_limit_seconds:
            wait_time = self.rate_limit_seconds - elapsed
            time.sleep(wait_time)

        self._last_request[domain] = time.time()

    def _is_placeholder_url(self, url: str) -> bool:
        """Check if URL looks like a placeholder image."""
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in self.PLACEHOLDER_PATTERNS)

    def _compute_md5(self, data: bytes) -> str:
        """Compute MD5 hash of image data."""
        return hashlib.md5(data).hexdigest()

    def _compute_perceptual_hash(self, image: Image.Image) -> Optional[str]:
        """Compute perceptual hash for image similarity detection."""
        if not IMAGEHASH_AVAILABLE:
            return None

        try:
            # Use average hash for speed, or phash for accuracy
            phash = imagehash.average_hash(image)
            return str(phash)
        except Exception as e:
            logger.warning(f"Failed to compute perceptual hash: {e}")
            return None

    def _generate_filename(self, url: str, article_id: str, format: str) -> str:
        """Generate a unique filename for the image."""
        # Use article_id as base, add URL hash for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
        extension = format.lower()
        if extension == "jpeg":
            extension = "jpg"
        return f"{article_id}_{url_hash}.{extension}"

    def _validate_image(
        self,
        image: Image.Image,
        file_size: int,
        url: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate image meets our criteria.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check dimensions
        width, height = image.size
        if width < self.min_width:
            errors.append(f"Width too small: {width}px (min: {self.min_width}px)")
        if height < self.min_height:
            errors.append(f"Height too small: {height}px (min: {self.min_height}px)")

        # Check file size
        if file_size > self.max_file_size_bytes:
            size_mb = file_size / (1024 * 1024)
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            errors.append(f"File too large: {size_mb:.1f}MB (max: {max_mb:.1f}MB)")

        # Check format
        img_format = image.format or "UNKNOWN"
        if img_format.upper() not in self.SUPPORTED_FORMATS:
            errors.append(f"Unsupported format: {img_format}")

        # Check for placeholder URL
        if self._is_placeholder_url(url):
            errors.append("URL appears to be a placeholder image")

        # Check for very small aspect ratios (likely logos/icons)
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            errors.append(f"Unusual aspect ratio: {aspect_ratio:.2f} (likely banner/logo)")

        return len(errors) == 0, errors

    def download(
        self,
        url: str,
        article_id: str,
        skip_duplicates: bool = True
    ) -> Optional[DownloadedImage]:
        """
        Download and validate an image from a URL.

        Args:
            url: URL of the image to download
            article_id: ID of the article this image belongs to
            skip_duplicates: Skip if image hash already seen

        Returns:
            DownloadedImage object or None if download/validation failed
        """
        if not url:
            return None

        domain = self._get_domain(url)
        self._wait_for_rate_limit(domain)

        try:
            # Download image
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers, timeout=self.timeout, stream=True)
            response.raise_for_status()

            # Read image data
            image_data = response.content
            file_size = len(image_data)

            # Compute MD5 hash
            md5_hash = self._compute_md5(image_data)

            # Check for duplicates by MD5
            if skip_duplicates and md5_hash in self._seen_hashes:
                logger.debug(f"Skipping duplicate image (MD5): {url[:50]}...")
                return None

            # Open and validate image
            from io import BytesIO
            image = Image.open(BytesIO(image_data))

            # Validate
            is_valid, errors = self._validate_image(image, file_size, url)

            # Compute perceptual hash
            perceptual_hash = self._compute_perceptual_hash(image)

            # Check for perceptual duplicates
            if skip_duplicates and perceptual_hash and perceptual_hash in self._seen_perceptual_hashes:
                logger.debug(f"Skipping duplicate image (perceptual): {url[:50]}...")
                return None

            # Generate filename and save
            img_format = image.format or "JPEG"
            filename = self._generate_filename(url, article_id, img_format)
            local_path = self.output_dir / filename

            # Convert to RGB if necessary (for JPEG saving)
            if image.mode in ("RGBA", "P") and img_format.upper() in ("JPEG", "JPG"):
                image = image.convert("RGB")

            # Save image
            image.save(local_path, format=img_format if img_format != "JPG" else "JPEG")

            # Track hashes
            self._seen_hashes.add(md5_hash)
            if perceptual_hash:
                self._seen_perceptual_hashes.add(perceptual_hash)

            # Create result object
            result = DownloadedImage(
                local_path=str(local_path),
                filename=filename,
                original_url=url,
                source_domain=domain,
                width=image.size[0],
                height=image.size[1],
                format=img_format,
                file_size_bytes=file_size,
                md5_hash=md5_hash,
                perceptual_hash=perceptual_hash,
                is_valid=is_valid,
                validation_errors=errors,
            )

            if is_valid:
                logger.debug(f"Downloaded valid image: {filename} ({image.size[0]}x{image.size[1]})")
            else:
                logger.debug(f"Downloaded invalid image: {filename} - {errors}")

            return result

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to download image from {url[:50]}...: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing image from {url[:50]}...: {e}")
            return None

    def download_best_image(
        self,
        image_urls: List[str],
        article_id: str
    ) -> Optional[DownloadedImage]:
        """
        Try to download the best available image from a list of URLs.

        Tries each URL in order until one succeeds validation.

        Args:
            image_urls: List of image URLs to try
            article_id: ID of the article

        Returns:
            DownloadedImage or None if all failed
        """
        for url in image_urls:
            if not url:
                continue

            result = self.download(url, article_id)

            if result and result.is_valid:
                return result

        # If no valid image found, return first downloaded (even if invalid)
        for url in image_urls:
            if not url:
                continue

            result = self.download(url, article_id, skip_duplicates=False)
            if result:
                return result

        return None

    def get_stats(self) -> Dict:
        """Get statistics about downloaded images."""
        return {
            "unique_images_by_md5": len(self._seen_hashes),
            "unique_images_by_perceptual": len(self._seen_perceptual_hashes),
            "output_directory": str(self.output_dir),
        }

    def clear_cache(self) -> None:
        """Clear the deduplication caches."""
        self._seen_hashes.clear()
        self._seen_perceptual_hashes.clear()
        logger.info("Cleared image deduplication caches")


if __name__ == "__main__":
    # Quick test
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("Testing Image Downloader...")
    print("=" * 50)

    # Create downloader with test directory
    downloader = ImageDownloader(
        output_dir="data/raw/images/test",
        min_width=200,  # Lower threshold for testing
        min_height=200,
    )

    # Test URLs (using public domain images)
    test_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/800px-Camponotus_flavomarginatus_ant.jpg",
    ]

    for url in test_urls:
        print(f"\nDownloading: {url[:60]}...")
        result = downloader.download(url, "TEST-001")

        if result:
            print(f"  Success: {result.filename}")
            print(f"  Size: {result.width}x{result.height}")
            print(f"  Format: {result.format}")
            print(f"  Valid: {result.is_valid}")
            if not result.is_valid:
                print(f"  Errors: {result.validation_errors}")
        else:
            print("  Failed to download")

    print(f"\nStats: {downloader.get_stats()}")
