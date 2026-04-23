"""
RSS Feed Scraper for MultiGuard
===============================

Scrapes news articles from RSS feeds without requiring API keys.
Uses newspaper3k for full article extraction.
"""

import asyncio
import hashlib
import os
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import feedparser
import requests
from newspaper import Article, ArticleException
from loguru import logger
from tqdm import tqdm

# Fix SSL certificates for macOS
try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
except ImportError:
    pass

# Disable SSL warnings for development (some feeds have certificate issues)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class ScrapedArticle:
    """Represents a scraped news article with metadata."""

    # Unique identifier
    article_id: str

    # Content
    title: str
    text: str
    summary: Optional[str] = None

    # Source information
    url: str = ""
    source_domain: str = ""
    source_name: str = ""

    # Media
    image_url: Optional[str] = None
    image_urls: List[str] = field(default_factory=list)

    # Metadata
    authors: List[str] = field(default_factory=list)
    publish_date: Optional[datetime] = None
    keywords: List[str] = field(default_factory=list)

    # Scraping metadata
    scrape_date: datetime = field(default_factory=datetime.now)
    scrape_method: str = "rss"
    feed_name: str = ""
    category: str = ""

    # Validation flags
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "article_id": self.article_id,
            "title": self.title,
            "text": self.text,
            "summary": self.summary,
            "url": self.url,
            "source_domain": self.source_domain,
            "source_name": self.source_name,
            "image_url": self.image_url,
            "image_urls": self.image_urls,
            "authors": self.authors,
            "publish_date": self.publish_date.isoformat() if self.publish_date else None,
            "keywords": self.keywords,
            "scrape_date": self.scrape_date.isoformat(),
            "scrape_method": self.scrape_method,
            "feed_name": self.feed_name,
            "category": self.category,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }


class RSSScraper:
    """
    Scrapes news articles from RSS feeds.

    Features:
    - Parses RSS/Atom feeds
    - Extracts full article content using newspaper3k
    - Downloads article images
    - Rate limiting to respect servers
    - Deduplication based on URL hash
    """

    # Default RSS feeds if none provided
    DEFAULT_FEEDS = [
        {"name": "BBC News", "url": "http://feeds.bbci.co.uk/news/rss.xml", "category": "general"},
        {"name": "BBC World", "url": "http://feeds.bbci.co.uk/news/world/rss.xml", "category": "world"},
        {"name": "NPR News", "url": "https://feeds.npr.org/1001/rss.xml", "category": "general"},
        {"name": "The Guardian World", "url": "https://www.theguardian.com/world/rss", "category": "world"},
        {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml", "category": "world"},
        {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/index", "category": "technology"},
        {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml", "category": "technology"},
        {"name": "Science Daily", "url": "https://www.sciencedaily.com/rss/all.xml", "category": "science"},
    ]

    def __init__(
        self,
        feeds: Optional[List[Dict[str, str]]] = None,
        rate_limit_seconds: float = 2.0,
        min_words: int = 100,
        user_agent: str = "MultiGuardBot/1.0 (academic research)"
    ):
        """
        Initialize the RSS scraper.

        Args:
            feeds: List of feed dictionaries with 'name', 'url', 'category'
            rate_limit_seconds: Minimum seconds between requests
            min_words: Minimum word count for valid articles
            user_agent: User agent string for requests
        """
        self.feeds = feeds or self.DEFAULT_FEEDS
        self.rate_limit_seconds = rate_limit_seconds
        self.min_words = min_words
        self.user_agent = user_agent

        # Track seen URLs to avoid duplicates
        self._seen_urls: set = set()

        # Track last request time per domain
        self._last_request: Dict[str, float] = {}

        logger.info(f"Initialized RSSScraper with {len(self.feeds)} feeds")

    def _generate_article_id(self, url: str) -> str:
        """Generate a unique ID for an article based on its URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"MG-{timestamp}-{url_hash.upper()}"

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")

    def _wait_for_rate_limit(self, domain: str) -> None:
        """Wait if necessary to respect rate limit for a domain."""
        now = time.time()
        last_request = self._last_request.get(domain, 0)
        elapsed = now - last_request

        if elapsed < self.rate_limit_seconds:
            wait_time = self.rate_limit_seconds - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.1f}s for {domain}")
            time.sleep(wait_time)

        self._last_request[domain] = time.time()

    def _extract_article_content(self, url: str) -> Optional[Article]:
        """
        Extract full article content from a URL using newspaper3k.

        Args:
            url: The article URL to extract content from

        Returns:
            newspaper.Article object or None if extraction failed
        """
        domain = self._get_domain(url)
        self._wait_for_rate_limit(domain)

        try:
            # First try to download with requests for better control
            try:
                response = requests.get(
                    url,
                    headers={"User-Agent": self.user_agent},
                    timeout=15,
                    verify=False
                )
                response.raise_for_status()
                html_content = response.text
            except requests.exceptions.RequestException as e:
                logger.debug(f"Requests failed for {url}, trying newspaper3k directly: {e}")
                html_content = None

            article = Article(url)
            article.config.browser_user_agent = self.user_agent
            article.config.request_timeout = 15

            if html_content:
                article.set_html(html_content)
            else:
                article.download()

            article.parse()

            return article

        except ArticleException as e:
            logger.warning(f"Failed to extract article from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting {url}: {e}")
            return None

    def _validate_article(self, article: ScrapedArticle) -> ScrapedArticle:
        """
        Validate an article meets our criteria.

        Args:
            article: The article to validate

        Returns:
            The article with validation flags set
        """
        errors = []

        # Check text length
        word_count = len(article.text.split()) if article.text else 0
        if word_count < self.min_words:
            errors.append(f"Text too short: {word_count} words (min: {self.min_words})")

        # Check title exists
        if not article.title or len(article.title.strip()) < 5:
            errors.append("Missing or too short title")

        # Check for image
        if not article.image_url and not article.image_urls:
            errors.append("No image found")

        article.validation_errors = errors
        article.is_valid = len(errors) == 0

        return article

    def scrape_feed(
        self,
        feed_url: str,
        feed_name: str = "",
        category: str = "",
        max_articles: int = 100
    ) -> List[ScrapedArticle]:
        """
        Scrape articles from a single RSS feed.

        Args:
            feed_url: URL of the RSS feed
            feed_name: Human-readable name of the feed
            category: Category of the feed (e.g., 'technology', 'science')
            max_articles: Maximum number of articles to scrape from this feed

        Returns:
            List of ScrapedArticle objects
        """
        logger.info(f"Scraping feed: {feed_name} ({feed_url})")

        try:
            # Fetch feed content with SSL handling
            try:
                response = requests.get(
                    feed_url,
                    headers={"User-Agent": self.user_agent},
                    timeout=30,
                    verify=False  # Skip SSL verification for problematic feeds
                )
                response.raise_for_status()
                feed_content = response.text
            except requests.exceptions.RequestException as e:
                logger.warning(f"HTTP request failed for {feed_name}, trying direct parse: {e}")
                feed_content = feed_url  # Fallback to letting feedparser handle it

            # Parse the RSS feed
            feed = feedparser.parse(feed_content)

            if feed.bozo:
                logger.warning(f"Feed parsing warning for {feed_name}: {feed.bozo_exception}")

            entries = feed.entries[:max_articles]
            logger.info(f"Found {len(entries)} entries in {feed_name}")

        except Exception as e:
            logger.error(f"Failed to parse feed {feed_name}: {e}")
            return []

        articles = []

        for entry in tqdm(entries, desc=f"Scraping {feed_name}", leave=False):
            # Get article URL
            url = entry.get("link", "")
            if not url:
                continue

            # Skip if already seen
            if url in self._seen_urls:
                logger.debug(f"Skipping duplicate URL: {url}")
                continue

            self._seen_urls.add(url)

            # Extract full article content
            newspaper_article = self._extract_article_content(url)

            if not newspaper_article:
                continue

            # Get publish date
            publish_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    publish_date = datetime(*entry.published_parsed[:6])
                except Exception:
                    pass

            # Create ScrapedArticle
            article = ScrapedArticle(
                article_id=self._generate_article_id(url),
                title=newspaper_article.title or entry.get("title", ""),
                text=newspaper_article.text or "",
                summary=entry.get("summary", newspaper_article.meta_description),
                url=url,
                source_domain=self._get_domain(url),
                source_name=feed_name,
                image_url=newspaper_article.top_image or None,
                image_urls=list(newspaper_article.images) if newspaper_article.images else [],
                authors=newspaper_article.authors or [],
                publish_date=publish_date,
                keywords=newspaper_article.keywords or [],
                feed_name=feed_name,
                category=category,
                scrape_method="rss",
            )

            # Validate
            article = self._validate_article(article)
            articles.append(article)

            if article.is_valid:
                logger.debug(f"Scraped valid article: {article.title[:50]}...")
            else:
                logger.debug(f"Invalid article: {article.validation_errors}")

        valid_count = sum(1 for a in articles if a.is_valid)
        logger.info(f"Scraped {len(articles)} articles from {feed_name} ({valid_count} valid)")

        return articles

    def scrape_all_feeds(
        self,
        max_articles_per_feed: int = 50,
        target_total: Optional[int] = None
    ) -> List[ScrapedArticle]:
        """
        Scrape articles from all configured RSS feeds.

        Args:
            max_articles_per_feed: Maximum articles to scrape per feed
            target_total: Stop when this many valid articles are collected (optional)

        Returns:
            List of all scraped articles
        """
        all_articles = []
        valid_count = 0

        logger.info(f"Starting to scrape {len(self.feeds)} feeds...")

        for feed_config in self.feeds:
            if target_total and valid_count >= target_total:
                logger.info(f"Reached target of {target_total} valid articles")
                break

            articles = self.scrape_feed(
                feed_url=feed_config["url"],
                feed_name=feed_config["name"],
                category=feed_config.get("category", "general"),
                max_articles=max_articles_per_feed
            )

            all_articles.extend(articles)
            valid_count = sum(1 for a in all_articles if a.is_valid)

            logger.info(f"Total progress: {valid_count} valid articles collected")

        logger.info(f"Finished scraping. Total: {len(all_articles)} articles, {valid_count} valid")

        return all_articles

    def get_valid_articles(self, articles: List[ScrapedArticle]) -> List[ScrapedArticle]:
        """Filter to only valid articles."""
        return [a for a in articles if a.is_valid]

    def get_articles_with_images(self, articles: List[ScrapedArticle]) -> List[ScrapedArticle]:
        """Filter to only articles that have images."""
        return [a for a in articles if a.image_url or a.image_urls]


# Convenience function for quick scraping
def scrape_rss_feeds(
    feeds: Optional[List[Dict[str, str]]] = None,
    max_per_feed: int = 50,
    target_total: Optional[int] = None
) -> List[ScrapedArticle]:
    """
    Quick function to scrape RSS feeds.

    Args:
        feeds: List of feed configs, or None for defaults
        max_per_feed: Max articles per feed
        target_total: Stop when this many valid articles collected

    Returns:
        List of ScrapedArticle objects
    """
    scraper = RSSScraper(feeds=feeds)
    return scraper.scrape_all_feeds(
        max_articles_per_feed=max_per_feed,
        target_total=target_total
    )


if __name__ == "__main__":
    # Quick test
    from loguru import logger
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("Testing RSS Scraper...")
    print("=" * 50)

    # Test with a single feed
    scraper = RSSScraper()
    articles = scraper.scrape_feed(
        feed_url="http://feeds.bbci.co.uk/news/rss.xml",
        feed_name="BBC News",
        category="general",
        max_articles=5
    )

    print(f"\nScraped {len(articles)} articles")

    for article in articles[:3]:
        print(f"\n- Title: {article.title[:60]}...")
        print(f"  URL: {article.url}")
        print(f"  Words: {len(article.text.split())}")
        print(f"  Image: {article.image_url[:50] if article.image_url else 'None'}...")
        print(f"  Valid: {article.is_valid}")
        if not article.is_valid:
            print(f"  Errors: {article.validation_errors}")
