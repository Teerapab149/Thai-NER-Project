import feedparser
import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
import re

# ---------- RSS SOURCES ----------
RSS_FEEDS = [
    "https://www.thairath.co.th/rss/news",
    "https://www.thairath.co.th/rss/politic",
    "https://www.thairath.co.th/rss/economy",
    "https://www.thairath.co.th/rss/sport",
    "https://www.matichon.co.th/rss/generalnews",
    "https://www.khaosod.co.th/rss/entertainment",
    "https://www.matichon.co.th/rss/news",
    "https://www.khaosod.co.th/rss/news",
    "https://www.posttoday.com/rss/news",
    "https://www.dailynews.co.th/rss/news",
    "https://www.sanook.com/news/rss/",
    "https://www.komchadluek.net/news/feed",
    "https://www.bangkokbiznews.com/rss/news",
    "https://www.thaipost.net/main/rss",
    "https://www.innnews.co.th/rss/news",
    "https://www.ryt9.com/rss/getnews.php?type=1",
    "https://www.springnews.co.th/rss/news",
    "https://www.amarintv.com/feed/news",
    "https://www.tnnthailand.com/rss/news",
    "https://www.thairath.co.th/rss/foreign",
    "https://www.thairath.co.th/rss/local",
    "https://www.khaosod.co.th/rss/foreign",
    "https://www.khaosod.co.th/rss/crime",
]

# ---------- SAVE PATH ----------
output_file = Path("thai_ner_project/data/news.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

# ---------- HELPERS ----------
def clean_html(raw_html: str) -> str:
    """ล้าง HTML, แท็ก, และช่องว่าง"""
    if not raw_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def fetch_full_article(url: str) -> str:
    """พยายามดึงเนื้อข่าวเต็ม ถ้าทำได้"""
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")

        # พยายามเลือก container หลักของข่าว
        possible_tags = [
            "article", "div[itemprop='articleBody']", "div.entry-content",
            "div.td-post-content", "div.main-content", "section.article"
        ]
        for tag in possible_tags:
            content = soup.select_one(tag)
            if content:
                text = clean_html(content.get_text(separator=" "))
                if len(text) > 200:
                    return text
        return ""
    except Exception:
        return ""

# ---------- MAIN ----------
all_articles = []
for feed_url in RSS_FEEDS:
    feed = feedparser.parse(feed_url)
    for entry in feed.entries[:30]:  # ดึงข่าวละไม่เกิน 10 ชิ้น
        text = ""
        # 1️⃣ ลองใช้ description ก่อน
        if hasattr(entry, "description"):
            text = clean_html(entry.description)
        # 2️⃣ ถ้ายังสั้นเกินไป ลองดึงเนื้อข่าวเต็ม
        if len(text) < 200 and hasattr(entry, "link"):
            full_text = fetch_full_article(entry.link)
            if len(full_text) > 200:
                text = full_text

        # ข้ามข่าวที่ไม่มีเนื้อหา
        if not text or len(text.split()) < 10:
            continue

        record = {
            "source": feed_url,
            "data": {"text": entry.title + "\n" + text},
            "meta": {"url": entry.link},
        }
        all_articles.append(record)
        print(f"✅ ดึงข่าวจาก {feed_url.split('//')[1].split('/')[0]} : {entry.title[:50]}...")

# ---------- SAVE ----------
with open(output_file, "w", encoding="utf-8") as f:
    for r in all_articles:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n📰 เก็บข่าวสำเร็จทั้งหมด {len(all_articles)} ข่าว -> {output_file}")
