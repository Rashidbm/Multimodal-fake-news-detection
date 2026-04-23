#!/usr/bin/env python3
"""News scraping CLI (RSS / APIs)."""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import click
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel

from src.scraping.news_scraper import NewsScraper


# Rich console for pretty output
console = Console()


def setup_logging(verbose: bool = False, log_file: str = None):
    """Configure logging."""
    logger.remove()

    level = "DEBUG" if verbose else "INFO"

    # Console logging
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    )

    # File logging
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
        )


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                      MultiGuard                           ║
║              News Scraping System v0.1.0                  ║
╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def print_stats(stats: dict):
    """Print scraping statistics in a nice table."""
    table = Table(title="Scraping Statistics", show_header=True, header_style="bold magenta")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Articles Scraped", str(stats.get("total_articles_scraped", 0)))
    table.add_row("Valid Articles", str(stats.get("valid_articles", 0)))
    table.add_row("Articles with Images", str(stats.get("articles_with_images", 0)))
    table.add_row("Images Downloaded", str(stats.get("images_downloaded", 0)))
    table.add_row("Valid Images", str(stats.get("valid_images", 0)))
    table.add_row("Complete Samples", str(stats.get("complete_samples", 0)))
    table.add_row("Failed Images", str(stats.get("failed_images", 0)))

    if stats.get("duration_seconds"):
        duration = stats["duration_seconds"]
        if duration > 3600:
            duration_str = f"{duration/3600:.1f} hours"
        elif duration > 60:
            duration_str = f"{duration/60:.1f} minutes"
        else:
            duration_str = f"{duration:.0f} seconds"
        table.add_row("Duration", duration_str)

    console.print(table)

    # Print by source
    if stats.get("by_source"):
        source_table = Table(title="Articles by Source", show_header=True, header_style="bold blue")
        source_table.add_column("Source", style="cyan")
        source_table.add_column("Count", style="green")

        for source, count in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
            source_table.add_row(source, str(count))

        console.print(source_table)


@click.command()
@click.option(
    "--source",
    "-s",
    type=click.Choice(["rss", "gnews", "newsapi", "all"]),
    default="rss",
    help="Data source to use. 'rss' requires no API keys.",
)
@click.option(
    "--count",
    "-c",
    type=int,
    default=100,
    help="Target number of complete articles to collect.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/raw",
    help="Output directory for scraped data.",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to settings.yaml config file.",
)
@click.option(
    "--no-images",
    is_flag=True,
    default=False,
    help="Skip image downloading (faster).",
)
@click.option(
    "--max-per-feed",
    type=int,
    default=50,
    help="Maximum articles to scrape per RSS feed.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from previous scraping session.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
@click.option(
    "--log-file",
    type=click.Path(),
    default="logs/scraping.log",
    help="Log file path.",
)
def main(
    source: str,
    count: int,
    output: str,
    config: str,
    no_images: bool,
    max_per_feed: int,
    resume: bool,
    verbose: bool,
    log_file: str,
):
    """
    Scrape news articles for the MultiGuard dataset.

    This tool collects news articles with images from various sources.
    By default, it uses RSS feeds which require no API keys.

    Examples:

        # Basic usage - scrape 100 articles using RSS
        python scripts/scrape_news.py --count 100

        # Scrape 1000 articles for dataset
        python scripts/scrape_news.py --count 1000 --source rss

        # Fast mode without images
        python scripts/scrape_news.py --count 500 --no-images

        # Resume interrupted session
        python scripts/scrape_news.py --count 1000 --resume
    """
    # Setup
    setup_logging(verbose, log_file)
    print_banner()

    # Show configuration
    console.print(Panel(
        f"[bold]Source:[/bold] {source}\n"
        f"[bold]Target:[/bold] {count} complete articles\n"
        f"[bold]Output:[/bold] {output}\n"
        f"[bold]Images:[/bold] {'Disabled' if no_images else 'Enabled'}\n"
        f"[bold]Resume:[/bold] {'Yes' if resume else 'No'}",
        title="Configuration",
        expand=False,
    ))

    # Determine config path
    if config is None:
        default_config = project_root / "config" / "settings.yaml"
        if default_config.exists():
            config = str(default_config)
            console.print(f"[dim]Using config: {config}[/dim]")

    # Initialize scraper
    console.print("\n[bold yellow]Initializing scraper...[/bold yellow]")

    try:
        scraper = NewsScraper(
            config_path=config,
            output_dir=output,
        )
    except Exception as e:
        console.print(f"[bold red]Failed to initialize scraper: {e}[/bold red]")
        sys.exit(1)

    # Show existing data if resuming
    if resume:
        existing = len(scraper._seen_article_ids)
        if existing > 0:
            console.print(f"[green]Found {existing} existing articles. Will skip duplicates.[/green]")
        else:
            console.print("[yellow]No existing articles found. Starting fresh.[/yellow]")

    # Start scraping
    console.print(f"\n[bold green]Starting scraping...[/bold green]")
    console.print("[dim]This may take a while. Press Ctrl+C to stop gracefully.[/dim]\n")

    try:
        # Determine sources
        sources = [source] if source != "all" else ["rss", "gnews", "newsapi"]

        # Run scraping
        collected = scraper.scrape(
            sources=sources,
            target_count=count,
            max_per_feed=max_per_feed,
            download_images=not no_images,
            save_incrementally=True,
        )

        # Print results
        console.print("\n[bold green]Scraping complete![/bold green]\n")
        print_stats(scraper.get_stats())

        # Summary
        complete = [c for c in collected if c.is_complete]
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total articles: {len(collected)}")
        console.print(f"  Complete samples (text + image): {len(complete)}")
        console.print(f"  Output directory: {output}")

        if len(complete) < count:
            console.print(
                f"\n[yellow]Note: Only collected {len(complete)}/{count} complete samples. "
                f"Run again with --resume to continue.[/yellow]"
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Scraping interrupted by user.[/yellow]")
        console.print("[dim]Data collected so far has been saved.[/dim]")
        print_stats(scraper.get_stats())
        sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]Error during scraping: {e}[/bold red]")
        logger.exception("Scraping error")
        sys.exit(1)


if __name__ == "__main__":
    main()
