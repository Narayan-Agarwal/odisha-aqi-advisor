"""Find WAQI station UIDs for Odisha cities."""
import requests, json, time

TOKEN = "ef5feda541fb94c36eb71257d7687ef36bc929ca"
KEYWORDS = ["jharsuguda", "angul", "talcher", "rourkela", "sambalpur",
            "bhubaneswar", "cuttack", "balasore", "brahmapur", "berhampur",
            "rayagada", "odisha", "orissa"]

for kw in KEYWORDS:
    r = requests.get("https://api.waqi.info/search/",
                     params={"token": TOKEN, "keyword": kw}, timeout=10)
    data = r.json()
    if data.get("status") == "ok" and data.get("data"):
        for s in data["data"][:5]:
            print(f"{kw}: uid={s['uid']} name={s['station']['name']}")
    else:
        print(f"{kw}: no results")
    time.sleep(0.4)
