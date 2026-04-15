

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. LOAD DATASET
# ──────────────────────────────────────────────

def load_dataset(path="styles.csv"):
    df = pd.read_csv(path, on_bad_lines="skip")
    df.dropna(subset=["articleType", "baseColour", "gender"], inplace=True)
    df["articleType"]       = df["articleType"].str.strip()
    df["baseColour"]        = df["baseColour"].str.strip()
    df["gender"]            = df["gender"].str.strip()
    df["productDisplayName"] = df["productDisplayName"].fillna("Unknown Product")
    return df

df = load_dataset()

# ──────────────────────────────────────────────
# 1. STATIC LOOKUP TABLES
# ──────────────────────────────────────────────

# Spelling / OCR / slang corrections  →  canonical query word
SPELLING_FIXES: dict[str, str] = {
    # pants / trousers
    "pents":    "pants",
    "pant":     "pants",
    "trouser":  "trousers",

    # shirts / tshirts
    "tshirt":   "tshirts",
    "t-shirt":  "tshirts",
    "t shirt":  "tshirts",
    "t shirts": "tshirts",
    "t-shirts": "tshirts",
    "tee":      "tshirts",
    "tees":     "tshirts",

    # accessories
    "watchs":   "watches",
    "wach":     "watches",
    "sunglass": "sunglasses",
    "goggle":   "sunglasses",
    "goggles":  "sunglasses",
    "purse":    "bags",
    "handbag":  "bags",
    "backpack": "bags",

    # footwear
    "sneaker":  "sneakers",
    "sneaker":  "sneakers",
    "boot":     "boots",
    "sandal":   "sandals",
    "slipper":  "slippers",
    "flip flop":"slippers",
    "flip-flop":"slippers",
    "loafer":   "shoes",
    "heel":     "heels",

    # tops
    "blouse":   "tops",
    "kurti":    "kurtas",

    # bottoms
    "jean":     "jeans",
    "short":    "shorts",
    "legging":  "leggings",
    "skirt":    "skirts",

    # dresses
    "gown":     "dresses",
    "frock":    "dresses",

    # accessories
    "belt":     "belts",
    "scarf":    "scarves",
    "hat":      "caps",
    "cap":      "caps",
    "sock":     "socks",
}

# Synonym map: canonical query word → list of dataset articleType values
SYNONYM_MAP: dict[str, list[str]] = {
    # topwear
    "tshirts":      ["Tshirts"],
    "shirts":       ["Shirts"],
    "tops":         ["Tops", "Shirts", "Tshirts"],
    "kurtas":       ["Kurtas", "Kurta Sets"],
    "sweaters":     ["Sweaters"],
    "jackets":      ["Jackets"],
    "hoodies":      ["Sweatshirts"],
    "sweatshirts":  ["Sweatshirts"],
    "blazers":      ["Blazers"],
    "coats":        ["Coats"],
    "suits":        ["Suits"],
    "tank":         ["Tops"],

    # bottomwear
    "pants":        ["Jeans", "Trousers", "Track Pants"],
    "jeans":        ["Jeans"],
    "trousers":     ["Trousers"],
    "shorts":       ["Shorts"],
    "leggings":     ["Leggings", "Tights"],
    "skirts":       ["Skirts"],
    "track pants":  ["Track Pants"],

    # dresses / ethnic
    "dresses":      ["Dresses"],
    "saree":        ["Sarees"],
    "salwar":       ["Salwar"],
    "dupatta":      ["Dupatta"],

    # footwear
    "shoes":        ["Casual Shoes", "Sports Shoes", "Formal Shoes"],
    "sneakers":     ["Sports Shoes", "Casual Shoes"],
    "boots":        ["Boots"],
    "sandals":      ["Sandals", "Flats"],
    "heels":        ["Heels", "Wedges"],
    "slippers":     ["Flip Flops"],
    "formal shoes": ["Formal Shoes"],
    "sports shoes": ["Sports Shoes"],
    "casual shoes": ["Casual Shoes"],

    # accessories
    "watches":      ["Watches"],
    "belts":        ["Belts"],
    "bags":         ["Handbags", "Backpacks", "Clutches", "Messenger Bag"],
    "sunglasses":   ["Sunglasses"],
    "caps":         ["Caps", "Hats"],
    "scarves":      ["Scarves"],
    "socks":        ["Socks"],
    "jewellery":    ["Jewellery"],
    "necklace":     ["Necklace and Chains"],
    "earrings":     ["Earrings"],
    "bracelet":     ["Bracelets"],
    "ring":         ["Ring"],

    # innerwear / activewear
    "innerwear":    ["Briefs", "Boxers", "Bra", "Innerwear Vests"],
    "sportswear":   ["Sports Shoes", "Tracksuits", "Sports Sandals"],
}

# Noise words to strip from user input
NOISE_WORDS = {
    "for", "the", "a", "an", "some", "any", "show", "me", "get",
    "give", "find", "search", "looking", "need", "want", "please",
    "in", "of", "with", "and", "or", "good", "nice", "best",
}

# Gender keyword → dataset value(s)
GENDER_MAP: dict[str, list[str]] = {
    "men":    ["Men"],
    "man":    ["Men"],
    "male":   ["Men"],
    "women":  ["Women"],
    "woman":  ["Women"],
    "female": ["Women"],
    "lady":   ["Women"],
    "ladies": ["Women"],
    "girl":   ["Girls"],
    "girls":  ["Girls"],
    "boy":    ["Boys"],
    "boys":   ["Boys"],
    "kid":    ["Boys", "Girls"],
    "kids":   ["Boys", "Girls"],
    "child":  ["Boys", "Girls"],
    "unisex": ["Unisex"],
}

# Secondary / complementary recommendations
COMPLEMENTS: dict[str, list[str]] = {
    "Tshirts":      ["Shirts", "Tops", "Jeans", "Shorts"],
    "Shirts":       ["Tshirts", "Tops", "Trousers", "Jeans"],
    "Tops":         ["Tshirts", "Shirts", "Skirts", "Leggings"],
    "Jeans":        ["Tshirts", "Shirts", "Casual Shoes"],
    "Trousers":     ["Shirts", "Formal Shoes"],
    "Shorts":       ["Tshirts", "Casual Shoes", "Sandals"],
    "Dresses":      ["Sandals", "Heels", "Handbags"],
    "Casual Shoes": ["Jeans", "Shorts", "Tshirts"],
    "Formal Shoes": ["Trousers", "Shirts"],
    "Handbags":     ["Dresses", "Tops"],
    "Watches":      ["Belts", "Wallets"],
    "Kurtas":       ["Trousers", "Leggings"],
}


# ──────────────────────────────────────────────
# 2. normalize_input()
# ──────────────────────────────────────────────

def normalize_input(raw: str) -> str:
    """
    Lowercase, strip, apply multi-word spelling fixes first,
    then single-word fixes, remove punctuation noise.
    """
    text = raw.lower().strip()

    # Apply multi-word fixes first (preserves "t shirt" → "tshirts")
    for wrong, correct in sorted(SPELLING_FIXES.items(), key=lambda x: -len(x[0])):
        text = text.replace(wrong, correct)

    # Remove extra punctuation except spaces
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ──────────────────────────────────────────────
# 3. parse_input()
# ──────────────────────────────────────────────

def parse_input(user_input: str) -> dict:
    """
    Extract: color, article_types (list), gender (list), season, usage.
    Returns a dict with these keys.
    """
    text = normalize_input(user_input)
    words = text.split()

    result = {
        "color":         None,
        "article_types": [],
        "gender":        None,
        "season":        None,
        "usage":         None,
        "raw_text":      text,
    }

    # ── Gender detection (full-word match to avoid false positives) ──
    for keyword, genders in GENDER_MAP.items():
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, text):
            result["gender"] = genders
            break

    # ── Color detection (dataset-driven, longest match first) ──
    all_colors = sorted(df["baseColour"].dropna().unique(), key=len, reverse=True)
    for color in all_colors:
        if re.search(r"\b" + re.escape(color.lower()) + r"\b", text):
            result["color"] = color
            break

    # ── Season detection ──
    for season in ["Summer", "Winter", "Fall", "Spring"]:
        if season.lower() in text:
            result["season"] = season
            break

    # ── Usage detection ──
    usage_keywords = {
        "casual":  "Casual",
        "formal":  "Formal",
        "sports":  "Sports",
        "ethnic":  "Ethnic",
        "party":   "Party",
        "travel":  "Travel",
        "outdoor": "Sports",
        "gym":     "Sports",
        "office":  "Formal",
    }
    for kw, usage_val in usage_keywords.items():
        if kw in text:
            result["usage"] = usage_val
            break

    # ── Article type detection ──
    # Build tokens from longest synonym key to shortest to catch phrases
    detected_types: list[str] = []

    # Phase 1: multi-word synonym keys
    for key in sorted(SYNONYM_MAP.keys(), key=len, reverse=True):
        if len(key.split()) > 1 and key in text:
            detected_types.extend(SYNONYM_MAP[key])

    # Phase 2: single-word token matching
    clean_words = [w for w in words if w not in NOISE_WORDS]
    for word in clean_words:
        if word in SYNONYM_MAP:
            detected_types.extend(SYNONYM_MAP[word])

    # Phase 3: fallback — direct articleType match from dataset
    if not detected_types:
        article_types_lower = {a.lower(): a for a in df["articleType"].unique()}
        for word in clean_words:
            if word in article_types_lower:
                detected_types.append(article_types_lower[word])

    # Deduplicate while preserving order
    seen: set[str] = set()
    for t in detected_types:
        if t not in seen:
            result["article_types"].append(t)
            seen.add(t)

    return result


# ──────────────────────────────────────────────
# 4. search_engine()
# ──────────────────────────────────────────────

def search_engine(
    user_input: str,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Core search function. Returns:
      primary   – best matching results
      secondary – complementary / similar items
      parsed    – the parsed query dict (for display)
    """
    parsed = parse_input(user_input)
    color         = parsed["color"]
    article_types = parsed["article_types"]
    gender        = parsed["gender"]
    season        = parsed["season"]
    usage         = parsed["usage"]

    base = df.copy()

    # ── Strict gender filter ──
    if gender:
        base = base[base["gender"].isin(gender)]

    # ── Optional season filter ──
    if season and "season" in base.columns:
        season_df = base[base["season"] == season]
        if not season_df.empty:
            base = season_df

    # ── Optional usage filter ──
    if usage and "usage" in base.columns:
        usage_df = base[base["usage"] == usage]
        if not usage_df.empty:
            base = usage_df

    # Nothing at all understood
    if not color and not article_types:
        return pd.DataFrame(), pd.DataFrame(), parsed

    # ── Build primary results ──
    if color and article_types:
        primary = base[
            (base["baseColour"] == color) &
            (base["articleType"].isin(article_types))
        ]
        # Relax color if too few results
        if len(primary) < top_n:
            fallback = base[base["articleType"].isin(article_types)]
            primary = pd.concat([primary, fallback]).drop_duplicates(subset="id")

    elif color:
        primary = base[base["baseColour"] == color]

    else:  # only article_types
        primary = base[base["articleType"].isin(article_types)]

    primary = primary.head(top_n)

    # ── Build secondary (complementary) results ──
    complement_types: list[str] = []
    for at in article_types:
        complement_types.extend(COMPLEMENTS.get(at, []))
    complement_types = list(dict.fromkeys(complement_types))  # dedup

    secondary = pd.DataFrame()
    if complement_types:
        sec_q = base[base["articleType"].isin(complement_types)]
        if color:
            sec_color = sec_q[sec_q["baseColour"] == color]
            if not sec_color.empty:
                sec_q = sec_color
        secondary = sec_q.head(top_n)

    return primary, secondary, parsed


# ──────────────────────────────────────────────
# 5. show_results()
# ──────────────────────────────────────────────

PLACEHOLDER_COLOR = (220, 220, 220)  # light grey

def _load_image(img_id: int) -> "np.ndarray | None":
    path = os.path.join("images", f"{img_id}.jpg")
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_results(
    title: str,
    data: pd.DataFrame,
    parsed: dict | None = None,
) -> None:
    """
    Renders a clean product grid with names and images.
    Falls back to a grey placeholder when image is missing.
    """
    import numpy as np

    if data.empty:
        print(f"\n⚠️  No results found for: {title}")
        return

    n = len(data)
    fig = plt.figure(figsize=(min(n * 3, 18), 5))

    # ── Header ──
    header_color = "#1a1a2e"
    fig.patch.set_facecolor(header_color)

    # Title text
    plt.suptitle(
        f"  {title}",
        fontsize=15,
        fontweight="bold",
        color="white",
        x=0.02,
        ha="left",
        y=1.02,
    )

    # Query info subtitle
    if parsed:
        parts = []
        if parsed.get("color"):        parts.append(f"Color: {parsed['color']}")
        if parsed.get("article_types"):parts.append(f"Type: {', '.join(parsed['article_types'])}")
        if parsed.get("gender"):       parts.append(f"Gender: {', '.join(parsed['gender'])}")
        if parts:
            plt.figtext(0.02, 0.97, "  " + " | ".join(parts),
                        fontsize=9, color="#aaaacc", ha="left")

    axes = []
    for i in range(n):
        ax = fig.add_subplot(1, n, i + 1)
        axes.append(ax)
        ax.set_facecolor(header_color)

    for i, (idx, row) in enumerate(data.iterrows()):
        ax = axes[i]
        img = _load_image(int(row["id"]))

        if img is not None:
            ax.imshow(img, aspect="auto")
        else:
            # Draw placeholder
            placeholder = np.full((300, 225, 3), PLACEHOLDER_COLOR, dtype=np.uint8)
            ax.imshow(placeholder)
            ax.text(
                0.5, 0.5, "No Image",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=9, color="#888888",
            )

        # Product name (wrap at ~20 chars)
        name = str(row.get("productDisplayName", ""))
        if len(name) > 22:
            name = name[:20] + "…"

        ax.set_title(
            f"{row['articleType']}\n{name}",
            fontsize=7.5,
            color="white",
            pad=4,
            wrap=True,
        )
        ax.axis("off")

    plt.tight_layout(pad=1.0)
    plt.show()


# ──────────────────────────────────────────────
# 6. run_system()  – main REPL loop
# ──────────────────────────────────────────────

_SEPARATOR = "─" * 52


def run_system(top_n: int = 5) -> None:
    """
    Interactive fashion search REPL.
    Type 'exit' or 'quit' to stop.
    """
    print("\n" + "═" * 52)
    print("   👗  HYBRID FASHION SEARCH ENGINE")
    print("   Powered by NLP + Smart Filtering")
    print("═" * 52)
    print("  Try: 'black tshirt for women', 'pants',")
    print("       'shoes for men', 'watchs', 'bags' …")
    print(_SEPARATOR)

    while True:
        try:
            user_input = input("\n🔍 Search: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Exiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("👋 Exiting. Goodbye!")
            break

        primary, secondary, parsed = search_engine(user_input, top_n=top_n)

        print("\n" + _SEPARATOR)
        print(f"  INPUT : {user_input}")
        print(f"  PARSED: color={parsed['color']} | "
              f"types={parsed['article_types']} | "
              f"gender={parsed['gender']}")
        print(_SEPARATOR)

        if primary.empty and secondary.empty:
            print("❌  Could not find any matching products.")
            print("   Tip: Try simpler terms like 'blue shirt' or 'women sandals'.")
            continue

        if not primary.empty:
            show_results("🎯 Top Matches", primary, parsed)
            # Print product list
            print("\n  Products found:")
            for _, row in primary.iterrows():
                print(f"    • [{row['gender']}] {row['articleType']} "
                      f"| {row['baseColour']} "
                      f"| {row.get('productDisplayName', '')[:50]}")
        else:
            print("  (No exact matches — showing complementary items only)")

        if not secondary.empty:
            show_results("💡 You Might Also Like", secondary, parsed)


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    run_system(top_n=5)
