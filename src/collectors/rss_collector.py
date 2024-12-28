import feedparser

import logging

class RSSCollector:
    """
    RSSCollector class is responsible for fetching and parsing RSS feed data 
    from a given URL. It takes the feed URL and language as input and 
    returns a list of parsed news entries.
    """

    def __init__(self, rss_url: str, language: str):
        """
        Constructor to initialize the RSSCollector with the feed URL and the language.
        
        :param rss_url: The URL of the RSS feed
        :param language: The language code/string of the content in the feed
        """
        self.rss_url = rss_url
        self.language = language

    def fetch_news(self):
        """
        Fetches the RSS feed data using the feedparser library and returns a list of news entries.
        
        :return: A list of dictionaries containing news entries
        """
        # Parse the RSS feed from the given URL
        feed = feedparser.parse(self.rss_url)

        # If the feed is not parsed correctly, log a warning and return an empty list
        if feed.bozo:
            logging.warning(f"Failed to parse RSS feed from {self.rss_url}.")
            return []

        news_entries = []
        # Loop through each entry in the parsed feed to structure the data in a standardized format
        for entry in feed.entries:
            # Some RSS feeds might not contain all fields, so we check their existence
            news_entry = {
                "title": entry.title if hasattr(entry, "title") else "",
                "description": entry.description if hasattr(entry, "description") else "",
                "summary": entry.summary if hasattr(entry, "summary") else "",
                "link": entry.link if hasattr(entry, "link") else "",
                "published": entry.published if hasattr(entry, "published") else "",
                # We try to detect language from the feed itself; if not present, fallback to self.language
                "language": entry.language if hasattr(entry, "language") else self.language,
                "news_text": entry.title + " " + entry.description + " " + entry.summary
            }
            news_entries.append(news_entry)

        return news_entries
