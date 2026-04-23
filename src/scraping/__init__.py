# Scraping module for MultiGuard
"""
News scraping components for collecting authentic news articles with images.
"""

from .rss_scraper import RSSScraper, ScrapedArticle, scrape_rss_feeds
from .image_downloader import ImageDownloader, DownloadedImage
from .news_scraper import NewsScraper, CollectedArticle, quick_scrape

__all__ = [
    "RSSScraper",
    "ScrapedArticle",
    "scrape_rss_feeds",
    "ImageDownloader",
    "DownloadedImage",
    "NewsScraper",
    "CollectedArticle",
    "quick_scrape",
]
