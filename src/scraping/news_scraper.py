"""
Main News Scraper Orchestrator for MultiGuard
==============================================

Coordinates scraping from multiple sources (RSS, APIs)
and manages the complete data collection pipeline.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
from loguru import logger
from tqdm import tqdm

from .rss_scraper import RSSScraper, ScrapedArticle
from .image_downloader import ImageDownloader, DownloadedImage


@dataclass
class CollectedArticle:
    """
    Represents a fully collected article with text and image.
    Ready for dataset processing.
    """

    # Article data
    article: ScrapedArticle

    # Downloaded image
    image: Optional[DownloadedImage] = None

    # Status
    has_valid_text: bool = False
    has_valid_image: bool = False
    is_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "article": self.article.to_dict(),
            "image": self.image.to_dict() if self.image else None,
            "has_valid_text": self.has_valid_text,
            "has_valid_image": self.has_valid_image,
            "is_complete": self.is_complete,
        }


@dataclass
class ScrapingStats:
    """Statistics about the scraping session."""

    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Counts
    total_articles_scraped: int = 0
    valid_articles: int = 0
    articles_with_images: int = 0
    images_downloaded: int = 0
    valid_images: int = 0
    complete_samples: int = 0

    # By source
    by_source: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)

    # Errors
    failed_articles: int = 0
    failed_images: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "total_articles_scraped": self.total_articles_scraped,
            "valid_articles": self.valid_articles,
            "articles_with_images": self.articles_with_images,
            "images_downloaded": self.images_downloaded,
            "valid_images": self.valid_images,
            "complete_samples": self.complete_samples,
            "by_source": self.by_source,
            "by_category": self.by_category,
            "failed_articles": self.failed_articles,
            "failed_images": self.failed_images,
        }


class NewsScraper:
    """
    Main orchestrator for news scraping.

    Coordinates:
    - RSS feed scraping
    - API-based scraping (when keys available)
    - Image downloading
    - Data persistence

    Features:
    - Resume interrupted scraping sessions
    - Progress tracking
    - Configurable from YAML
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "data/raw",
        rate_limit_seconds: float = 2.0,
        min_words: int = 100,
    ):
        """
        Initialize the news scraper.

        Args:
            config_path: Path to settings.yaml (optional)
            output_dir: Base directory for output
            rate_limit_seconds: Delay between requests
            min_words: Minimum words for valid articles
        """
        self.output_dir = Path(output_dir)
        self.articles_dir = self.output_dir / "articles"
        self.images_dir = self.output_dir / "images"
        self.rate_limit_seconds = rate_limit_seconds
        self.min_words = min_words

        # Create directories
        self.articles_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Load config if provided
        self.config = self._load_config(config_path) if config_path else {}

        # Initialize sub-scrapers
        self.rss_scraper = RSSScraper(
            feeds=self.config.get("rss_feeds"),
            rate_limit_seconds=rate_limit_seconds,
            min_words=min_words,
        )

        self.image_downloader = ImageDownloader(
            output_dir=str(self.images_dir),
            min_width=self.config.get("image", {}).get("min_width", 400),
            min_height=self.config.get("image", {}).get("min_height", 300),
            rate_limit_seconds=1.0,
        )

        # Track collected data
        self.collected: List[CollectedArticle] = []
        self.stats = ScrapingStats()

        # Track seen article IDs for resume
        self._seen_article_ids: set = set()
        self._load_existing_articles()

        logger.info(f"Initialized NewsScraper. Output: {self.output_dir}")
        logger.info(f"Found {len(self._seen_article_ids)} existing articles")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded config from {config_path}")
                return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}

    def _load_existing_articles(self) -> None:
        """Load IDs of already scraped articles for resume functionality."""
        for json_file in self.articles_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    article_id = data.get("article", {}).get("article_id")
                    if article_id:
                        self._seen_article_ids.add(article_id)
            except Exception:
                pass

    def _save_article(self, collected: CollectedArticle) -> None:
        """Save a collected article to disk."""
        filename = f"{collected.article.article_id}.json"
        filepath = self.articles_dir / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(collected.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved article: {filename}")
        except Exception as e:
            logger.error(f"Failed to save article {filename}: {e}")

    def _save_stats(self) -> None:
        """Save scraping statistics."""
        filepath = self.output_dir / "scraping_stats.json"
        try:
            with open(filepath, "w") as f:
                json.dump(self.stats.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    def scrape_rss(
        self,
        target_count: int = 1000,
        max_per_feed: int = 100,
        download_images: bool = True,
        save_incrementally: bool = True,
    ) -> List[CollectedArticle]:
        """
        Scrape articles from RSS feeds.

        Args:
            target_count: Target number of complete articles to collect
            max_per_feed: Maximum articles per RSS feed
            download_images: Whether to download images
            save_incrementally: Save each article as it's collected

        Returns:
            List of CollectedArticle objects
        """
        logger.info(f"Starting RSS scraping. Target: {target_count} complete articles")

        # Scrape articles from RSS
        articles = self.rss_scraper.scrape_all_feeds(
            max_articles_per_feed=max_per_feed,
            target_total=target_count * 2,  # Scrape extra to account for filtering
        )

        self.stats.total_articles_scraped = len(articles)
        logger.info(f"Scraped {len(articles)} articles from RSS feeds")

        # Process each article
        complete_count = 0

        for article in tqdm(articles, desc="Processing articles"):
            # Skip if already collected
            if article.article_id in self._seen_article_ids:
                logger.debug(f"Skipping already collected: {article.article_id}")
                continue

            # Create collected article
            collected = CollectedArticle(article=article)

            # Check text validity
            collected.has_valid_text = article.is_valid

            if collected.has_valid_text:
                self.stats.valid_articles += 1

            # Download image if available
            if download_images and (article.image_url or article.image_urls):
                self.stats.articles_with_images += 1

                # Try primary image first, then others
                image_urls = []
                if article.image_url:
                    image_urls.append(article.image_url)
                image_urls.extend(article.image_urls)

                downloaded = self.image_downloader.download_best_image(
                    image_urls=image_urls,
                    article_id=article.article_id,
                )

                if downloaded:
                    collected.image = downloaded
                    self.stats.images_downloaded += 1

                    if downloaded.is_valid:
                        collected.has_valid_image = True
                        self.stats.valid_images += 1
                else:
                    self.stats.failed_images += 1

            # Check if complete (valid text + valid image)
            collected.is_complete = collected.has_valid_text and collected.has_valid_image

            if collected.is_complete:
                complete_count += 1
                self.stats.complete_samples += 1

            # Track by source
            source = article.source_name or "unknown"
            self.stats.by_source[source] = self.stats.by_source.get(source, 0) + 1

            # Track by category
            category = article.category or "general"
            self.stats.by_category[category] = self.stats.by_category.get(category, 0) + 1

            # Add to collection
            self.collected.append(collected)
            self._seen_article_ids.add(article.article_id)

            # Save incrementally
            if save_incrementally:
                self._save_article(collected)

            # Check if target reached
            if complete_count >= target_count:
                logger.info(f"Reached target of {target_count} complete articles")
                break

        # Finalize stats
        self.stats.end_time = datetime.now()
        self._save_stats()

        logger.info(f"Scraping complete. {complete_count} complete samples collected")
        logger.info(f"Stats: {self.stats.to_dict()}")

        return self.collected

    def scrape(
        self,
        sources: List[str] = None,
        target_count: int = 1000,
        **kwargs
    ) -> List[CollectedArticle]:
        """
        Main scraping method. Coordinates all sources.

        Args:
            sources: List of sources to use ['rss', 'gnews', 'newsapi']
            target_count: Target number of complete samples
            **kwargs: Additional arguments passed to source scrapers

        Returns:
            List of CollectedArticle objects
        """
        sources = sources or ["rss"]

        all_collected = []

        for source in sources:
            if source == "rss":
                collected = self.scrape_rss(
                    target_count=target_count - len(all_collected),
                    **kwargs
                )
                all_collected.extend(collected)

            elif source == "gnews":
                logger.warning("GNews API scraping not yet implemented")
                # TODO: Implement GNews API scraping

            elif source == "newsapi":
                logger.warning("NewsAPI scraping not yet implemented")
                # TODO: Implement NewsAPI scraping

            else:
                logger.warning(f"Unknown source: {source}")

            # Check if target reached
            complete = [c for c in all_collected if c.is_complete]
            if len(complete) >= target_count:
                break

        return all_collected

    def get_complete_articles(self) -> List[CollectedArticle]:
        """Get only complete articles (valid text + valid image)."""
        return [c for c in self.collected if c.is_complete]

    def get_stats(self) -> Dict:
        """Get current statistics."""
        return self.stats.to_dict()

    def load_collected_articles(self) -> List[CollectedArticle]:
        """Load all previously collected articles from disk."""
        collected = []

        for json_file in self.articles_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Reconstruct ScrapedArticle
                article_data = data.get("article", {})
                article = ScrapedArticle(
                    article_id=article_data.get("article_id", ""),
                    title=article_data.get("title", ""),
                    text=article_data.get("text", ""),
                    summary=article_data.get("summary"),
                    url=article_data.get("url", ""),
                    source_domain=article_data.get("source_domain", ""),
                    source_name=article_data.get("source_name", ""),
                    image_url=article_data.get("image_url"),
                    image_urls=article_data.get("image_urls", []),
                    authors=article_data.get("authors", []),
                    keywords=article_data.get("keywords", []),
                    feed_name=article_data.get("feed_name", ""),
                    category=article_data.get("category", ""),
                    is_valid=article_data.get("is_valid", True),
                    validation_errors=article_data.get("validation_errors", []),
                )

                # Reconstruct DownloadedImage if present
                image = None
                image_data = data.get("image")
                if image_data:
                    image = DownloadedImage(
                        local_path=image_data.get("local_path", ""),
                        filename=image_data.get("filename", ""),
                        original_url=image_data.get("original_url", ""),
                        source_domain=image_data.get("source_domain", ""),
                        width=image_data.get("width", 0),
                        height=image_data.get("height", 0),
                        format=image_data.get("format", ""),
                        file_size_bytes=image_data.get("file_size_bytes", 0),
                        md5_hash=image_data.get("md5_hash", ""),
                        perceptual_hash=image_data.get("perceptual_hash"),
                        is_valid=image_data.get("is_valid", True),
                        validation_errors=image_data.get("validation_errors", []),
                    )

                collected_article = CollectedArticle(
                    article=article,
                    image=image,
                    has_valid_text=data.get("has_valid_text", False),
                    has_valid_image=data.get("has_valid_image", False),
                    is_complete=data.get("is_complete", False),
                )

                collected.append(collected_article)

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(collected)} articles from disk")
        return collected


def quick_scrape(
    target_count: int = 100,
    output_dir: str = "data/raw",
    sources: List[str] = None,
) -> List[CollectedArticle]:
    """
    Quick function to scrape news articles.

    Args:
        target_count: Number of complete articles to collect
        output_dir: Output directory
        sources: Sources to use (default: RSS only)

    Returns:
        List of CollectedArticle objects
    """
    scraper = NewsScraper(output_dir=output_dir)
    return scraper.scrape(
        sources=sources or ["rss"],
        target_count=target_count,
    )


if __name__ == "__main__":
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("Testing News Scraper...")
    print("=" * 50)

    # Quick test with small target
    scraper = NewsScraper(output_dir="data/raw")

    # Scrape a few articles
    collected = scraper.scrape_rss(
        target_count=5,
        max_per_feed=10,
        download_images=True,
    )

    print(f"\nCollected {len(collected)} articles")
    complete = [c for c in collected if c.is_complete]
    print(f"Complete samples: {len(complete)}")

    print("\nSample articles:")
    for c in complete[:3]:
        print(f"\n- {c.article.title[:60]}...")
        print(f"  Source: {c.article.source_name}")
        print(f"  Words: {len(c.article.text.split())}")
        if c.image:
            print(f"  Image: {c.image.filename} ({c.image.width}x{c.image.height})")

    print(f"\nStats: {scraper.get_stats()}")
