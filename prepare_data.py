import requests
from bs4 import BeautifulSoup
from pythainlp import word_tokenize
import json
from pathlib import Path
import time

# โฟลเดอร์เก็บข้อมูล
out_dir = Path("thai_ner_project/data")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "news.jsonl"

# RSS feed ที่จะดึงข่าว (เพิ่มได้เรื่อย ๆ)
RSS_FEEDS = [
    "https://www.thairath.co.th/rss/news"
]

def fetch_rss_links(feed_url):
    """ดึงลิงก์ข่าวจาก RSS (รองรับหลายรูปแบบ)"""
    try:
        r = requests.get(feed_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "xml")

        links = []
        for item in soup.find_all("item"):
            link_tag = item.find("link")
            if link_tag:
                # ดึงทั้ง text กับ .string เผื่อรูปแบบต่างกัน
                link = link_tag.text.strip() if link_tag.text else link_tag.string.strip()
                links.append(link)

        # fallback ถ้าไม่มี <item>
        if not links:
            for link_tag in soup.find_all("link"):
                href = link_tag.text.strip() if link_tag.text else ""
                if href.startswith("http"):
                    links.append(href)

        return list(set(links))  # กันซ้ำ
    except Exception as e:
        print(f"[error] ดึง RSS ไม่ได้จาก {feed_url} -> {e}")
        return []

def fetch_article_text(url):
    """ดึงหัวข้อข่าว (<h1>) + เนื้อหาข่าว (<p>)"""
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # 🔹 หัวข้อข่าว
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # 🔹 รวมเนื้อหาข่าวจากทุกแท็ก <p>
        paragraphs = soup.find_all("p")
        content = "\n".join(p.get_text(strip=True) for p in paragraphs)

        # 🔹 รวมทั้งหมดเป็นข้อความเดียว (หัวข้อ + เนื้อหา)
        full_text = (title + "\n" + content).strip()

        # 🔹 กรองข่าวที่ไม่มีเนื้อหา
        if len(content) < 50:
            print(f"[skip] เนื้อหาสั้นเกินไป {url}")
            return None

        return full_text

    except Exception as e:
        print(f"[skip] {url} → {e}")
        return None

    finally:
        print(f"[done] โหลดเสร็จ: {url}")

def main():
    all_news = []
    for feed in RSS_FEEDS:
        print(f"กำลังดึงจาก: {feed}")
        links = fetch_rss_links(feed)
        print(f"  เจอลิงก์ {len(links)} ข่าว")

        for url in links[:5]:  # จำกัดข่าวละ 5 ชิ้น (ลองก่อน)
            text = fetch_article_text(url)
            if not text:
                continue

            # ตัดคำ
            tokens = word_tokenize(text, engine="newmm")

            # เก็บเป็น JSONL (พร้อมสำหรับ Label Studio)
            record = {
                "data": {"text": text, "tokens": tokens},
                "meta": {"url": url}
            }
            all_news.append(record)
            print(f"  ✅ เก็บข่าวจาก {url}")
            time.sleep(1)  # หน่วงเวลานิดกันโดน block

    # เขียนไฟล์ออกมา
    with open(out_file, "w", encoding="utf-8") as f:
        for news in all_news:
            f.write(json.dumps(news, ensure_ascii=False) + "\n")

    print(f"\nบันทึกข่าวทั้งหมด {len(all_news)} ชิ้น -> {out_file}")

if __name__ == "__main__":
    main()
