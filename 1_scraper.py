import os
import json
import time
import sys
import re
import shutil
from datetime import datetime
import pandas as pd
import praw
from dotenv import load_dotenv

def print_status(message: str):
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    safe_msg = message[:max(width - 1, 0)]
    sys.stdout.write(f"\x1b[2K\r{safe_msg}")
    sys.stdout.flush()

load_dotenv()
with open("config.json", "r") as f:
    config = json.load(f)

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

os.makedirs("raw_data", exist_ok=True)
rules = config["filtering_rules"]
target_years = config["scraping_settings"]["target_years"]

domain_name = config["domain_name"]

seen_post_ids = set()
seen_text_fingerprints = set()
global_saved_by_year = {year: 0 for year in target_years}

for year in target_years:
    fpath = f"raw_data/{domain_name}_{year}.csv"
    if os.path.exists(fpath):
        try:
            existing_df = pd.read_csv(fpath)
            if 'post_id' in existing_df.columns:
                seen_post_ids.update(existing_df['post_id'].astype(str).tolist())
            if 'body_text' in existing_df.columns:
                for text in existing_df['body_text'].dropna():
                    clean_fp = re.sub(r'\W+', '', text[:200].lower())
                    seen_text_fingerprints.add(clean_fp)
        except Exception:
            pass

print("\n STARTING REDDIT DATA COLLECTION...")
total_subs = len(config["target_subreddits"])

for idx, sub_name in enumerate(config["target_subreddits"], 1):
    sub_saved_by_year = {year: 0 for year in target_years}
    scanned_count = 0
    
    base_msg = f"  [{idx}/{total_subs}] Scanning r/{sub_name:22} "
    print(base_msg, end="", flush=True)
    blink_toggle = True 
    
    try:
        subreddit = reddit.subreddit(sub_name)
        feeds = [
            subreddit.new(limit=400),
            subreddit.hot(limit=400),
            subreddit.top(time_filter="all", limit=800)
        ]
        
        for feed in feeds:
            for post in feed:
                try:
                    scanned_count += 1
                
                    marker = "." if blink_toggle else " "
                    print_status(f"{base_msg}[Scanned: {scanned_count} | {marker}]")
                    blink_toggle = not blink_toggle

                    if post.id in seen_post_ids:
                        continue
                    author_name = str(post.author).lower() if post.author else ""
                    if author_name in ["automoderator", "moderator"] or "bot" in author_name:
                        continue
                        
                    if not post.selftext or len(post.selftext) < rules["min_character_length"]:
                        continue
                    if post.num_comments < rules["min_comments"]:
                        continue
                    if post.upvote_ratio < rules["min_upvote_ratio"]:
                        continue
                        

                    text_fingerprint = re.sub(r'\W+', '', post.selftext[:200].lower())
                    if text_fingerprint in seen_text_fingerprints:
                        continue
                        
                    post_year = datetime.fromtimestamp(post.created_utc).year
                    if post_year not in target_years:
                        continue
                    
                    post_data = {
                        "post_id": post.id,
                        "subreddit": sub_name,
                        "title": post.title,
                        "body_text": post.selftext,
                        "comments": post.num_comments,
                        "upvote_ratio": post.upvote_ratio,
                        "created_year": post_year
                    }
                    
                    seen_post_ids.add(post.id)
                    seen_text_fingerprints.add(text_fingerprint)
                    sub_saved_by_year[post_year] += 1
                    global_saved_by_year[post_year] += 1
                    
                    file_path = f"raw_data/{domain_name}_{post_year}.csv"
                    df = pd.DataFrame([post_data])
                    file_exists = os.path.exists(file_path)
                    df.to_csv(file_path, mode='a', header=not file_exists, index=False)
                    
                    time.sleep(config["scraping_settings"]["time_delay_seconds"])
                    
                except Exception:
                    continue
                    
        stats_str = ", ".join([f"{y}: {c} posts" for y, c in sub_saved_by_year.items()])
        sys.stdout.write(f"\x1b[2K\r ✅ [{idx}/{total_subs}] r/{sub_name:22} Done! Total Scanned: {scanned_count:<4} | Saved -> {stats_str}\n")
        sys.stdout.flush()
            
    except Exception:
        sys.stdout.write(f"\x1b[2K\r ⚠️  [{idx}/{total_subs}] r/{sub_name:22} Skipped or rate-limited. Moving on...\n")
        sys.stdout.flush()
        time.sleep(5) 

print(f"\n ALL SUBREDDITS COMPLETED! Session totals stored: {global_saved_by_year}\n")