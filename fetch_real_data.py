import json
import os
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "data"

def main():
    print("Loading Cornell's Hate Speech & Offensive Language dataset...")
    # class mapping in this dataset:
    # 0 - hate speech
    # 1 - offensive language (harassment/general toxicity)
    # 2 - neither (clean)
    ds = load_dataset("hate_speech_offensive", split="train")

    # Group by class
    hate = ds.filter(lambda x: x["class"] == 0).shuffle(seed=42).select(range(10))
    offensive = ds.filter(lambda x: x["class"] == 1).shuffle(seed=42).select(range(10))
    clean = ds.filter(lambda x: x["class"] == 2).shuffle(seed=42).select(range(10))

    new_posts = []
    
    def process(rows, label, action):
        for row in rows:
            new_posts.append({
                "content_id": f"real_post_{row['count']}_{row['class']}",
                "content_type": "text_post",
                "text": row["tweet"],
                "author_region": "GLOBAL",
                "author_history": [],
                "language": "en",
                "ground_truth_labels": label,
                "ground_truth_action": action,
                "difficulty": "extreme"
            })

    process(hate, ["hate_speech"], "remove")
    process(offensive, ["harassment"], "remove")
    process(clean, ["clean"], "approve")

    # Load existing posts
    posts_path = DATA_DIR / "posts.json"
    if posts_path.exists():
        with open(posts_path, "r", encoding="utf-8") as f:
            existing_posts = json.load(f)
    else:
        existing_posts = []

    # Append new posts
    all_posts = existing_posts + new_posts

    with open(posts_path, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, indent=2, ensure_ascii=False)

    print(f"Successfully integrated {len(new_posts)} real-world X/Twitter posts into posts.json.")

if __name__ == "__main__":
    main()
