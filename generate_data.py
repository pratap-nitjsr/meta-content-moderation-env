import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

posts = [
  {
    "content_id": "post_001",
    "content_type": "text_post",
    "text": "Just had the best coffee of my life at the new cafe downtown! Highly recommend. ☕💯",
    "author_region": "US", "author_history": [], "language": "en",
    "ground_truth_labels": ["clean"], "ground_truth_action": "approve", "difficulty": "easy"
  },
  {
    "content_id": "post_002",
    "content_type": "text_post",
    "text": "P3opl3 from [ethnic group] should not be allowed in this country. They ruin absolutely everything fr fr. 😡😡",
    "author_region": "DE", "author_history": ["hate_speech"], "language": "en",
    "ground_truth_labels": ["hate_speech"], "ground_truth_action": "remove", "difficulty": "medium"
  },
  {
    "content_id": "post_003",
    "content_type": "text_post",
    "text": "CLICK HERE NOW!! You've won $10,000. Send your bank details to claim your prize!!! 💸👇 http://bit.ly/sc4mlink",
    "author_region": "NG", "author_history": ["spam", "scam"], "language": "en",
    "ground_truth_labels": ["spam", "scam"], "ground_truth_action": "remove", "difficulty": "medium"
  },
  {
    "content_id": "post_004",
    "content_type": "text_post",
    "text": "यह लोग हमारे देश के दुश्मन हैं, इन्हें बाहर निकालो और मार डालो। 🩸",
    "author_region": "IN", "author_history": [], "language": "hi",
    "ground_truth_labels": ["hate_speech", "violence"], "ground_truth_action": "remove", "difficulty": "hard"
  }
]

clean_texts = [
    "Beautiful day! the weather is kinda bussin no cap fr fr 🚶‍♂️", 
    "Love this movie. anyone going to the concert? @livenation wya?? 🔥🔥🔥", 
    "Great recipe for pasta, tbh I just eyeball the garlic🧄 ¯\\_(ツ)_/¯", 
    "Check out my new dog! #goldenretriever softest boi alive", 
    "gm world let's get this bread today. stay grinding 💪", 
    "Just vibing hbu? 👀", 
    "what a beautiful day! ☀️💖"
]
hate_texts = [
    "I h8 all of them tbh. [ethnic group] needs 2 be deported ASAP. go back 2 ur country 😡", 
    "th3y r literally 4nimals and sh0uld be exterminated ngl.", 
    "worst kind of people on this planet. gas them all fr.", 
    "w3 must p.u.r.g.e them completely."
]
hate_hi_texts = [
    "सबको खत्म कर दो। इन कीड़ों को जला दो सालो को! 🔪🩸"
]
harassment_texts = [
    "You are so ugly and stupid. krill urself bro", 
    "literally nobody likes u, unfollow yourself 🤡", 
    "Stop posting, u absolute r-tard.", 
    "I know where u live. don't sleep tonight.", 
    "u r a total l0ser, no cap. literal trash."
]
spam_texts = [
    "B.U.Y cheap R0lex watches here!!! bit.ly/spamurl123 🤑⌚", 
    "FREE FOLLOWERS! CLICK MY BIO NOW!!! 💥🎉 #freemoney #followers", 
    "Earn $5000 a week from home! DM me 'CASH' to start today!!!", 
    "90% DISCOUNT on ALL items! Store closing down! click link below 👇👇", 
    "Get crypto fast! send 0.1 ETH to 0xABCD and get 1.0 ETH back! HURRY"
]
misinfo_texts = [
    "WAKE UP SHEEPLE! The new vax contains 5G microchips designed by Bill Gates to track u. DO YOUR RESEARCH!! 🚫💉", 
    "The earth is objectively flat, NASA stands for Never A Straight Answer. Stop believing the simulation!!", 
    "Drinking pure bleach + lemon juice completely cures all viruses overnight. Big pharma is hiding this!!", 
    "5G towers literally emit frequencies that mutate your DNA and cause COVID. look at the bird deaths!"
]
violence_texts = [
    "I will literally chop ur head off and mail it to ur mom.", 
    "let's go hurt someone tonight, im bored. who's down to jump that kid?", 
    "Blood will spill. I've had enough of this fkn school.", 
    "K1lling them is the ONLY way out of this mess."
]
scam_texts = [
    "Urgent: Your IRS tax payment is overdue. Pay $500 in Target Gift Cards immediately or face arrest. 🛑", 
    "Your PayPal account is locked! Click here: http://secure-login-paypal.xyz to verify your identity.", 
    "I am Prince AlWaleed, I need a trustworthy partner to transfer $10M. Reply with your bank routing details.", 
    "Send me $50 on CashApp and I'll flip it to $500 for you in 1 hour. GUARANTEED."
]
self_harm_texts = [
    "I just can't take it anymore, goodbye world. this is my last tweet. 💔🔫", 
    "how many advil pills does it take to finally end it? asking for a friend"
]
multi_texts = [
    "You people are absolute trash and I will destroy u all. ugly [slur] rtards, go jump off a bridge.", 
    "Listen here u fgt, I will slice your throat and upload the vid. 🩸"
]

dist = []
cc = 5

for t in clean_texts:
    dist.append({"content_id": f"post_{cc:03d}", "content_type": "text_post", "text": t, "author_region": "US", "author_history": [], "language": "en", "ground_truth_labels": ["clean"], "ground_truth_action": "approve", "difficulty": "easy"})
    cc += 1

for t in hate_texts:
    dist.append({"content_id": f"post_{cc:03d}", "content_type": "text_post", "text": t, "author_region": "US", "author_history": [], "language": "en", "ground_truth_labels": ["hate_speech"], "ground_truth_action": "remove", "difficulty": "medium"})
    cc += 1

for t in hate_hi_texts:
    dist.append({"content_id": f"post_{cc:03d}", "content_type": "text_post", "text": t, "author_region": "IN", "author_history": [], "language": "hi", "ground_truth_labels": ["hate_speech"], "ground_truth_action": "remove", "difficulty": "hard"})
    cc += 1

for t in harassment_texts:
    dist.append({"content_id": f"post_{cc:03d}", "content_type": "text_post", "text": t, "author_region": "US", "author_history": [], "language": "en", "ground_truth_labels": ["harassment"], "ground_truth_action": "remove", "difficulty": "medium"})
    cc += 1

for t in spam_texts:
    dist.append({"content_id": f"post_{cc:03d}", "content_type": "text_post", "text": t, "author_region": "US", "author_history": [], "language": "en", "ground_truth_labels": ["spam"], "ground_truth_action": "remove", "difficulty": "medium"})
    cc += 1

for t in misinfo_texts:
    dist.append({"content_id": f"post_{cc:03d}", "content_type": "text_post", "text": t, "author_region": "US", "author_history": [], "language": "en", "ground_truth_labels": ["misinformation"], "ground_truth_action": "remove", "difficulty": "hard"})
    cc += 1

for t in violence_texts:
    dist.append({"content_id": f"post_{cc:03d}", "content_type": "text_post", "text": t, "author_region": "US", "author_history": [], "language": "en", "ground_truth_labels": ["violence"], "ground_truth_action": "remove", "difficulty": "medium"})
    cc += 1

for t in [scam_texts[0], scam_texts[1], scam_texts[2]]:
    dist.append({"content_id": f"post_{cc:03d}", "content_type": "text_post", "text": t, "author_region": "US", "author_history": [], "language": "en", "ground_truth_labels": ["scam"], "ground_truth_action": "remove", "difficulty": "medium"})
    cc += 1
    
dist.append({"content_id": f"post_{cc:03d}", "content_type": "text_post", "text": scam_texts[3], "author_region": "US", "author_history": [], "language": "en", "ground_truth_labels": ["scam"], "ground_truth_action": "remove", "difficulty": "medium"})
cc += 1

for t in self_harm_texts:
    dist.append({"content_id": f"post_{cc:03d}", "content_type": "text_post", "text": t, "author_region": "US", "author_history": [], "language": "en", "ground_truth_labels": ["self_harm"], "ground_truth_action": "remove", "difficulty": "hard"})
    cc += 1

for t in multi_texts:
    dist.append({"content_id": f"post_{cc:03d}", "content_type": "text_post", "text": t, "author_region": "US", "author_history": [], "language": "en", "ground_truth_labels": ["hate_speech", "harassment", "violence"], "ground_truth_action": "remove", "difficulty": "hard"})
    cc += 1

all_posts = posts + dist

with open(DATA_DIR / "posts.json", "w", encoding="utf-8") as f:
    json.dump(all_posts, f, indent=2, ensure_ascii=False)


# IMAGE DESCRIPTIONS (20 total) 
images = [
    {"content_id": "img_001", "content_type": "image_description", "text": "IMAGE VISUAL: A crowded street protest at night. Prominent sign in foreground reads: 'DEPORT ALL [slur] NOW!'. The sign has blood-red painted text. OCR EXTRACT: 'DEPORT ALL [slur] NOW!'", "author_region": "US", "author_history": [], "language": "en", "ground_truth_labels": ["hate_speech"], "ground_truth_action": "remove", "difficulty": "hard", "media_urls": ["https://picsum.photos/seed/img001/800/600"], "media_types": ["image"]}
]
c = 2
def add_img(lbls, text):
    global c
    images.append({"content_id": f"img_{c:03d}", "content_type": "image_description", "text": text, "author_region": "US", "author_history": [], "language": "en", "ground_truth_labels": lbls, "ground_truth_action": "approve" if lbls==["clean"] else "remove", "difficulty": "hard", "media_urls": [f"https://picsum.photos/seed/img{c:03d}/800/600"], "media_types": ["image"]})
    c+=1

for _ in range(5): add_img(["clean"], "IMAGE VISUAL: A golden retriever playing in a sunny dog park, catching a frisbee. High resolution, bright lighting. No text present.")
for _ in range(3): add_img(["hate_speech"], "IMAGE VISUAL: Graffiti sprayed on a brick wall showing a large dark swastika symbol next to the letters '1488'. OCR EXTRACT: '1488'")
for _ in range(3): add_img(["violence"], "IMAGE VISUAL: Grainy security camera footage screenshot. Two men in hooded jackets are engaging in a brutal fight. One man is holding a large machete above his head preparing to strike. Blood is visible on the pavement.")
for _ in range(3): add_img(["nudity"], "IMAGE VISUAL: Explicit webcam screenshot showing a person fully exposing their uncovered genitalia. The image is brightly lit in a bedroom setting.")
for _ in range(3): add_img(["misinformation"], "IMAGE VISUAL: A crudely edited bar chart titled 'VACCINE DEATHS 2024'. The bars exponentially increase. Bold red text overlaid says 'THEY ARE LYING TO YOU. IT ALTERS YOUR DNA.' OCR EXTRACT: 'VACCINE DEATHS 2024 THEY ARE LYING TO YOU. IT ALTERS YOUR DNA.'")
for _ in range(2): add_img(["violence", "self_harm"], "IMAGE VISUAL: First-person view of an arm covered in fresh, bleeding deep cuts from a razor blade. A bloody razor blade sits on the bathroom counter in the background.")

with open(DATA_DIR / "image_descriptions.json", "w", encoding="utf-8") as f:
    json.dump(images, f, indent=2)

# AD COPIES (20) 
ads = [
    {
      "content_id": "ad_001",
      "content_type": "ad_copy",
      "text": "🚨 LOSE 30 POUNDS IN 10 DAYS 🚨 with our patented miracle pill! Doctors hate us! Guaranteed results or 100% money back! Click here -> bit.ly/dietscam",
      "author_region": "US",
      "author_history": [],
      "language": "en",
      "ad_targeting": {"age_min": 18, "age_max": 65, "interests": ["weight loss"]},
      "ground_truth_labels": ["misinformation"],
      "ground_truth_action": "remove",
      "violated_policies": ["AD_POLICY_HEALTH_CLAIMS_001"],
      "difficulty": "hard", 
      "media_urls": ["https://picsum.photos/seed/ad001/800/400"], 
      "media_types": ["image"]
    }
]
c = 2
def add_ad(lbls, txt, targets, vp):
    global c
    ads.append({"content_id": f"ad_{c:03d}", "content_type": "ad_copy", "text": txt, "author_region": "US", "author_history": [], "language": "en", "ad_targeting": targets, "ground_truth_labels": lbls, "ground_truth_action": "approve" if lbls==["clean"] else "remove", "violated_policies": vp, "difficulty": "hard", "media_urls": [f"https://picsum.photos/seed/ad{c:03d}/800/400"], "media_types": ["image"]})
    c+=1

for _ in range(5): add_ad(["clean"], "Step into comfort with our new orthopedic sneakers! Designed for all-day walking. Shop the summer sale now for 20% off. 👟👞", {"age_min": 18}, [])
for _ in range(3): add_ad(["misinformation"], "CURE CANCER NATURALLY! Big pharma doesn't want you to know about this exotic root juice that aggressively targets tumors. Read the secret report here. 🌱🔬", {"age_min": 40}, ["AD_POLICY_HEALTH_CLAIMS_001"])
for _ in range(3): add_ad(["scam"], "💰 GUARANTEED 500% ROI IN 24 HOURS! Our AI crypto trading bot never loses. Deposit $100 and withdraw $500 tomorrow. Limited spots available! 💰🚀", {"age_min": 18}, ["AD_POLICY_FINANCIAL_001"])
for _ in range(3): add_ad(["hate_speech"], "Are they taking over your town? Keep your neighborhood PURE from [ethnic minority gang]. Join the local militia network today to protect your heritage. 🛡️✊", {"interests": ["nationalism"]}, ["AD_POLICY_TARGETING_001"])
for _ in range(3): add_ad(["violence"], "Ghost guns and unregistered AR-15 parts strictly for 'airsoft'. 🤫 No background checks required, ships discreetly in 2 parts. DM to order. 🔫📦", {"age_min": 18}, ["AD_POLICY_WEAPONS_001"])
for _ in range(2): add_ad(["scam", "misinformation"], "The US Government has SECRETS funds allocated to pay off your credit card debt, but you must act NOW. Click here to use this weird health trick loopholes to claim your $10,000 cash grant! 🏦💵", {"age_min": 50}, ["AD_POLICY_FINANCIAL_001", "AD_POLICY_HEALTH_CLAIMS_001"])

with open(DATA_DIR / "ad_copies.json", "w", encoding="utf-8") as f:
    json.dump(ads, f, indent=2)

# WHATSAPP THREADS (15)
threads = [
    {
      "thread_id": "thread_001",
      "messages": [
        {
          "content_id": "wa_001_m1",
          "content_type": "whatsapp_message",
          "text": "Hey did u hear?? The g0v is putting 5G microchips in the new covid boosters!! DO NOT TAKE IT! 🚫💉",
          "author_region": "BR", "author_history": [], "language": "pt", "position": 1
        },
        {
          "content_id": "wa_001_m2",
          "content_type": "whatsapp_message",
          "text": "OMG fr?? Yes I'm forwarding this to the family group chat rn!!",
          "author_region": "BR", "author_history": [], "language": "pt", "position": 2
        }
      ],
      "ground_truth_thread_labels": ["misinformation"],
      "ground_truth_action": "restrict",
      "difficulty": "extreme",
      "cultural_context": "Brazilian WhatsApp viral forwarding chains often spread deeply embedded anti-vax conspiracies disguised as friendly warnings."
    }
]
c = 2
def add_thread(lbls, context, msgs, act="remove"):
    global c
    t = {"thread_id": f"thread_{c:03d}", "messages": [], "ground_truth_thread_labels": lbls, "ground_truth_action": act, "difficulty": "extreme", "cultural_context": context}
    for i, m in enumerate(msgs):
        t["messages"].append({"content_id": f"wa_{c:03d}_m{i+1}", "content_type": "whatsapp_message", "text": m, "author_region": "US", "author_history": [], "language": "en", "position": i+1})
    threads.append(t)
    c+=1

for _ in range(3): add_thread(["clean"], "A harmless everyday family group chat logistics discussion.", ["<Voice Message: 0:13> Can someone pick up milk?", "Yeah I'll grab it on my way home rn."], "approve")
for _ in range(2): add_thread(["misinformation"], "A QAnon style conspiracy theory spreading rapidly between friends.", ["Bro the 5G towers are LITERALLY causing covid. I have proof.", "Send the link!! I knew the govt was lying."], "restrict")
for _ in range(3): add_thread(["scam"], "A pig-butchering romance/crypto scam unfolding in a thread.", ["Hello dear, my assistant mistakenly added your number. but you seem like a kind person 🌹", "Haha who is this? sure nice to meet u", "I am an investor in Dubai. I can show you how to double your money in crypto easily, just click http://scam.ru/wallet"], "remove")
for _ in range(3): add_thread(["harassment"], "A coordinated cyberbullying attempt among high school students.", ["Did u see that TikTok Sarah posted? She is so fkn fat and ugly lmaooo 💀🐷", "omg yes. let's all spam her comments with 🤮"], "remove")
for _ in range(3): add_thread(["clean"], "Aggressive but protected political discourse criticizing public officials.", ["The Mayor's new tax plan is absolutely r-tarded and destroys the middle class.", "Agreed. We need to vote that idiot out immediately and protest outside city hall."], "approve")

with open(DATA_DIR / "whatsapp_threads.json", "w", encoding="utf-8") as f:
    json.dump(threads, f, indent=2)

print("Data generation complete.")
