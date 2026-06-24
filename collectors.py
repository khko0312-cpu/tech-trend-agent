import feedparser
import hashlib
import httpx
from datetime import datetime

RSS_SOURCES = {
    "aws":   "https://aws.amazon.com/blogs/aws/feed/",
    "gcp":   "https://cloudblog.withgoogle.com/rss/",
    "azure": "https://azure.microsoft.com/en-us/blog/feed/",
}

HN_KEYWORDS = ["aws", "cloud", "llm", "ai", "gpt", "kubernetes", "serverless"]


def collect_rss(source_name, url, limit=10):
    feed = feedparser.parse(url)
    results = []
    for entry in feed.entries[:limit]:
        results.append({
            "id":        hashlib.md5(entry.link.encode()).hexdigest(),
            "source":    source_name,
            "title":     entry.title,
            "url":       entry.link,
            "published": entry.get("published", datetime.now().isoformat()),
            "summary":   entry.get("summary", ""),
        })
    return results


def collect_hacker_news(limit=30):
    try:
        top_ids = httpx.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10
        ).json()[:limit]
    except Exception as e:
        print(f"Hacker News 수집 실패: {e}")
        return []

    results = []
    for item_id in top_ids:
        try:
            item = httpx.get(
                f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json", timeout=10
            ).json()
        except Exception:
            continue
        title = (item.get("title") or "").lower()
        if any(kw in title for kw in HN_KEYWORDS):
            results.append({
                "id":        hashlib.md5(str(item_id).encode()).hexdigest(),
                "source":    "hackernews",
                "title":     item.get("title", ""),
                "url":       item.get("url", f"https://news.ycombinator.com/item?id={item_id}"),
                "published": datetime.now().isoformat(),
                "summary":   "",
            })
    return results


def collect_all(limit_per_source=10):
    all_articles = []
    for name, url in RSS_SOURCES.items():
        articles = collect_rss(name, url, limit=limit_per_source)
        print(f"[{name}] {len(articles)}건 수집")
        all_articles += articles

    hn_articles = collect_hacker_news(limit=30)
    print(f"[hackernews] {len(hn_articles)}건 수집 (필터링 후)")
    all_articles += hn_articles

    return all_articles


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    articles = collect_all()
    print(f"\n총 {len(articles)}건 수집 완료\n")
    for a in articles:
        print(f"[{a['source']}] {a['title']}")