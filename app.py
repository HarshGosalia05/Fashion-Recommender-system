# """
# 👕 AI Fashion Recommender System
# Streamlit web app converted from main.ipynb
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# from PIL import Image
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # ─────────────────────────────────────────────
# # PAGE CONFIG
# # ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="AI Fashion Recommender",
#     page_icon="👕",
#     layout="wide",
# )

# # ─────────────────────────────────────────────
# # CUSTOM CSS  – clean card-style UI
# # ─────────────────────────────────────────────
# st.markdown(
#     """
#     <style>
#     /* ── global ── */
#     body { font-family: 'Segoe UI', sans-serif; }

#     /* ── header band ── */
#     .header-band {
#         background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
#         padding: 2rem 2rem 1.5rem 2rem;
#         border-radius: 12px;
#         margin-bottom: 1.5rem;
#         text-align: center;
#     }
#     .header-band h1 { color: #e94560; font-size: 2.4rem; margin: 0; }
#     .header-band p  { color: #a8b2d8; font-size: 1rem; margin-top: 0.4rem; }

#     /* ── section title ── */
#     .section-title {
#         font-size: 1.25rem;
#         font-weight: 700;
#         color: #e94560;
#         border-left: 4px solid #e94560;
#         padding-left: 0.6rem;
#         margin: 1.4rem 0 0.8rem 0;
#     }

#     /* ── product card ── */
#     .product-card {
#         background: #1e2235;
#         border-radius: 10px;
#         padding: 0.6rem;
#         text-align: center;
#         margin-bottom: 0.6rem;
#         border: 1px solid #2a2f4e;
#         transition: transform 0.2s;
#     }
#     .product-card:hover { transform: translateY(-3px); border-color: #e94560; }
#     .product-name {
#         font-size: 0.78rem;
#         font-weight: 600;
#         color: #c9d1d9;
#         margin-top: 0.4rem;
#         overflow: hidden;
#         text-overflow: ellipsis;
#         display: -webkit-box;
#         -webkit-line-clamp: 2;
#         -webkit-box-orient: vertical;
#     }
#     .product-tag {
#         display: inline-block;
#         background: #e94560;
#         color: white;
#         font-size: 0.65rem;
#         padding: 2px 8px;
#         border-radius: 20px;
#         margin-top: 0.3rem;
#     }

#     /* ── no-result box ── */
#     .no-result {
#         background: #2d2d44;
#         border: 1px dashed #e94560;
#         border-radius: 8px;
#         padding: 1.5rem;
#         text-align: center;
#         color: #a8b2d8;
#         font-size: 1rem;
#     }

#     /* ── search bar override ── */
#     div[data-testid="stTextInput"] > div > div > input {
#         border-radius: 8px;
#         border: 2px solid #0f3460;
#         background: #16213e;
#         color: #e0e0e0;
#         padding: 0.6rem 1rem;
#     }
#     div[data-testid="stTextInput"] > div > div > input:focus {
#         border-color: #e94560;
#         box-shadow: 0 0 0 3px rgba(233,69,96,0.2);
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # ─────────────────────────────────────────────
# # DATA LOADING  (cached so it runs once)
# # ─────────────────────────────────────────────
# @st.cache_data(show_spinner="Loading fashion dataset…")
# def load_data():
#     df = pd.read_csv("styles.csv", on_bad_lines="skip")
#     df = df.dropna(subset=["usage", "baseColour", "season", "productDisplayName"])
#     df["year"] = df["year"].fillna(0).astype(int)
#     df = df.reset_index(drop=True)
#     return df


# @st.cache_resource(show_spinner="Building TF-IDF similarity index…")
# def build_tfidf(df):
#     df["text"] = df["articleType"] + " " + df["baseColour"] + " " + df["usage"]
#     tfidf = TfidfVectorizer(stop_words="english")
#     matrix = tfidf.fit_transform(df["text"])
#     sim = cosine_similarity(matrix, matrix)
#     return sim


# # ─────────────────────────────────────────────
# # ITEM MAP  (same as notebook)
# # ─────────────────────────────────────────────
# ITEM_MAP = {
#     "tshirt": ["Tshirts"],
#     "tshirts": ["Tshirts"],
#     "shirt": ["Shirts"],
#     "shirts": ["Shirts"],
#     "top": ["Tops"],
#     "tops": ["Tops"],
#     "pants": ["Jeans", "Trousers", "Track Pants"],
#     "jeans": ["Jeans"],
#     "trouser": ["Trousers"],
#     "trousers": ["Trousers"],
#     "shorts": ["Shorts"],
#     "short": ["Shorts"],
#     "shoe": ["Casual Shoes"],
#     "shoes": ["Casual Shoes"],
#     "sandals": ["Sandals"],
#     "watch": ["Watches"],
#     "watches": ["Watches"],
#     "belt": ["Belts"],
#     "belts": ["Belts"],
#     "bag": ["Handbags"],
#     "bags": ["Handbags"],
#     "sunglasses": ["Sunglasses"],
#     "dress": ["Dresses"],
#     "dresses": ["Dresses"],
#     "skirt": ["Skirts"],
#     "skirts": ["Skirts"],
# }

# # ─────────────────────────────────────────────
# # OUTFIT SUGGESTION MAP
# # Items shown when user searches for each article type
# # ─────────────────────────────────────────────
# OUTFIT_MAP = {
#     "Tshirts": ["Jeans", "Trousers", "Shorts"],
#     "Shirts": ["Jeans", "Trousers", "Shorts"],
#     "Tops": ["Jeans", "Trousers", "Skirts"],
#     "Dresses": ["Casual Shoes", "Sandals", "Belts"],
#     "Skirts": ["Tops", "Shirts", "Casual Shoes"],
#     "Jeans": ["Tshirts", "Shirts"],
#     "Trousers": ["Shirts", "Tshirts"],
#     "Track Pants": ["Tshirts"],
#     "Shorts": ["Tshirts", "Shirts"],
#     "Casual Shoes": ["Jeans", "Shorts", "Trousers"],
#     "Sandals": ["Dresses", "Shorts", "Skirts"],
#     "Watches": ["Shirts", "Tshirts"],
#     "Belts": ["Jeans", "Trousers"],
#     "Handbags": ["Dresses", "Tops"],
#     "Sunglasses": ["Tshirts", "Shirts", "Dresses"],
# }


# # ─────────────────────────────────────────────
# # CORE LOGIC  (reused from notebook)
# # ─────────────────────────────────────────────
# def normalize_input(user_input: str) -> str:
#     """Spelling normalisation — same logic as notebook."""
#     user_input = user_input.lower().strip()
#     words = user_input.split()
#     new_words = []
#     for w in words:
#         if w in ("pents", "pant"):
#             new_words.append("pants")
#         elif w == "watchs":
#             new_words.append("watches")
#         elif w in ("t-shirt", "tshirt", "tshirts", "t"):
#             new_words.append("tshirts")
#         else:
#             new_words.append(w)
#     return " ".join(new_words)


# def parse_input(user_input: str, df: pd.DataFrame):
#     """
#     Returns (color, items_list, gender)
#     Same logic as notebook.
#     """
#     user_input = normalize_input(user_input)
#     words = user_input.split()

#     color = None
#     items = []
#     gender = None

#     # Colour detection
#     for c in df["baseColour"].dropna().unique():
#         if c.lower() in user_input:
#             color = c
#             break

#     # Item detection via map
#     for w in words:
#         if w in ITEM_MAP:
#             items.extend(ITEM_MAP[w])

#     # Fallback: direct match against dataset article types
#     if not items:
#         for art in df["articleType"].unique():
#             if art.lower() in user_input:
#                 items.append(art)

#     # Gender detection (order matters: women before men)
#     if "women" in user_input or "woman" in user_input:
#         gender = "Women"
#     elif "men" in user_input or "man" in user_input:
#         gender = "Men"
#     elif "girl" in user_input:
#         gender = "Girls"
#     elif "boy" in user_input:
#         gender = "Boys"

#     return color, list(set(items)), gender


# def smart_recommend_full(
#     user_input: str,
#     df: pd.DataFrame,
#     top_n: int = 10,
#     gender_override: str = "All",
#     subcategory_override: str = "All",
# ) -> pd.DataFrame | None:
#     """
#     Main recommendation engine — same logic as notebook.
#     Extra args allow optional UI filter overrides.
#     """
#     color, items, gender = parse_input(user_input, df)

#     # If UI override is set, it wins
#     if gender_override != "All":
#         gender = gender_override

#     data = df.copy()

#     # Gender filter
#     if gender:
#         data = data[data["gender"] == gender]

#     # Sub-category (masterCategory) filter from UI dropdown
#     if subcategory_override != "All":
#         data = data[data["subCategory"] == subcategory_override]

#     # Item type filter (strict)
#     if items:
#         data = data[data["articleType"].isin(items)]

#     # Colour filter
#     if color:
#         data = data[data["baseColour"] == color]

#     if data.empty:
#         return None

#     return data.head(top_n)


# def get_outfit_recommendations(items: list, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame | None:
#     """
#     Suggest complementary items based on OUTFIT_MAP.
#     """
#     outfit_types = []
#     for art in items:
#         outfit_types.extend(OUTFIT_MAP.get(art, []))
#     outfit_types = list(set(outfit_types) - set(items))

#     if not outfit_types:
#         return None

#     outfit_data = df[df["articleType"].isin(outfit_types)]
#     if outfit_data.empty:
#         return None
#     return outfit_data.sample(min(top_n, len(outfit_data)))


# def get_similar_items(index: int, cosine_sim: np.ndarray, df: pd.DataFrame, top_n: int = 5):
#     """TF-IDF cosine-similarity based 'similar items'."""
#     if index >= len(cosine_sim):
#         return None
#     sim_scores = sorted(enumerate(cosine_sim[index]), key=lambda x: x[1], reverse=True)
#     sim_scores = [s for s in sim_scores[1:] if s[0] < len(df)][:top_n]
#     indices = [i[0] for i in sim_scores]
#     return df.iloc[indices]


# # ─────────────────────────────────────────────
# # IMAGE HELPER
# # ─────────────────────────────────────────────
# def load_image(product_id, size=(200, 260)):
#     """Load product image; return PIL Image or None."""
#     path = os.path.join("images", f"{product_id}.jpg")
#     if os.path.exists(path):
#         try:
#             img = Image.open(path).convert("RGB")
#             img.thumbnail(size, Image.LANCZOS)
#             return img
#         except Exception:
#             return None
#     return None


# def render_product_card(col, row):
#     """Render a single product card inside a Streamlit column."""
#     img = load_image(row["id"])
#     with col:
#         if img:
#             st.image(img, use_container_width=True)
#         else:
#             st.markdown(
#                 "<div style='height:160px;background:#2a2f4e;border-radius:8px;"
#                 "display:flex;align-items:center;justify-content:center;"
#                 "color:#555;font-size:0.75rem;'>No Image</div>",
#                 unsafe_allow_html=True,
#             )
#         st.markdown(
#             f"""
#             <div class="product-card">
#                 <div class="product-name">{row['productDisplayName']}</div>
#                 <span class="product-tag">{row['articleType']}</span>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )


# def render_product_grid(results: pd.DataFrame, cols_per_row: int = 5):
#     """Render a grid of product cards."""
#     rows = [results.iloc[i : i + cols_per_row] for i in range(0, len(results), cols_per_row)]
#     for row_df in rows:
#         cols = st.columns(cols_per_row)
#         for col, (_, product) in zip(cols, row_df.iterrows()):
#             render_product_card(col, product)


# # ─────────────────────────────────────────────
# # MAIN APP
# # ─────────────────────────────────────────────
# def main():
#     # ── Header ──
#     st.markdown(
#         """
#         <div class="header-band">
#             <h1>👕 AI Fashion Recommender System</h1>
#             <p>Search outfits using text &nbsp;|&nbsp;
#             Try: <em>"black shirt"</em> &nbsp;·&nbsp;
#             <em>"white tshirts for women"</em> &nbsp;·&nbsp;
#             <em>"blue pants for men"</em></p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     # ── Load data ──
#     with st.spinner("Initialising…"):
#         df = load_data()
#         cosine_sim = build_tfidf(df)

#     # ── Search row ──
#     col_input, col_btn = st.columns([5, 1])
#     with col_input:
#         user_query = st.text_input(
#             "",
#             placeholder='e.g.  "black shirt for men"  or  "white tshirts for women"',
#             label_visibility="collapsed",
#         )
#     with col_btn:
#         search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")

#     # ── Filters row ──
#     st.markdown("<br>", unsafe_allow_html=True)
#     f1, f2, f3 = st.columns(3)
#     with f1:
#         gender_filter = st.selectbox(
#             "👤 Gender",
#             ["All", "Men", "Women", "Boys", "Girls", "Unisex"],
#             index=0,
#         )
#     with f2:
#         sub_options = ["All"] + sorted(df["subCategory"].dropna().unique().tolist())
#         sub_filter = st.selectbox("🏷️ Sub-category", sub_options, index=0)
#     with f3:
#         top_n = st.slider("📦 Results per section", min_value=5, max_value=20, value=10, step=5)

#     st.divider()

#     # ── Run search ──
#     if search_clicked or user_query:
#         if not user_query.strip():
#             st.warning("⚠️ Please enter a search query first.")
#             return

#         with st.spinner("Searching…"):
#             results = smart_recommend_full(
#                 user_query,
#                 df,
#                 top_n=top_n,
#                 gender_override=gender_filter,
#                 subcategory_override=sub_filter,
#             )

#         # ── TOP RESULTS ──
#         st.markdown('<div class="section-title">✅ Top Results</div>', unsafe_allow_html=True)
#         if results is None or results.empty:
#             st.markdown(
#                 '<div class="no-result">😕 No results found. Try different keywords, '
#                 'e.g. <em>"red shirt"</em> or <em>"blue jeans for men"</em>.</div>',
#                 unsafe_allow_html=True,
#             )
#             return

#         st.caption(f"Found **{len(results)}** product(s) for: *{user_query}*")
#         render_product_grid(results, cols_per_row=5)

#         # ── OUTFIT SUGGESTIONS ──
#         found_types = results["articleType"].unique().tolist()
#         outfit_results = get_outfit_recommendations(found_types, df, top_n=top_n)

#         if outfit_results is not None and not outfit_results.empty:
#             st.markdown('<div class="section-title">👗 Outfit Suggestions</div>', unsafe_allow_html=True)
#             render_product_grid(outfit_results, cols_per_row=5)

#         # ── SIMILAR ITEMS (TF-IDF) ──
#         first_idx = results.index[0]
#         similar = get_similar_items(first_idx, cosine_sim, df, top_n=5)

#         if similar is not None and not similar.empty:
#             st.markdown('<div class="section-title">🔁 Similar Items</div>', unsafe_allow_html=True)
#             render_product_grid(similar, cols_per_row=5)

#     else:
#         # ── Landing placeholder ──
#         st.markdown(
#             """
#             <div style="text-align:center; color:#555; margin-top:3rem;">
#                 <div style="font-size:4rem;">👗</div>
#                 <p style="font-size:1.1rem; color:#888;">
#                     Type something in the search box above and click <strong>Search</strong>
#                 </p>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )


# if __name__ == "__main__":
#     main()


















"""
👕 AI Fashion Recommender System
────────────────────────────────
Version 2.0 – Improved & Fixed
Changes over v1:
  • get_outfit_recommendations() – gender/colour/usage-aware, no random sampling
  • Gender consistency across all three sections (Top Results, Outfits, Similar)
  • Sub-category dropdown built purely from dataset values (no hardcoding)
  • Smarter OUTFIT_MAP – strictly relevant pairs only
  • Premium card UI: gender badge + colour dot + usage label
  • Responsive grid, enhanced hover, better empty states
  • Section icons, spacing, loading spinner, landing hero chips
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Fashion Recommender",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════
# COLOUR PALETTE  – maps baseColour → hex for card dot badge
# ══════════════════════════════════════════════════════════════
COLOUR_HEX: dict[str, str] = {
    "Black": "#1a1a1a",        "White": "#f5f5f5",
    "Navy Blue": "#001f5b",    "Blue": "#1565c0",
    "Grey": "#757575",         "Red": "#c62828",
    "Green": "#2e7d32",        "Yellow": "#f9a825",
    "Pink": "#e91e8c",         "Orange": "#e65100",
    "Purple": "#6a1b9a",       "Brown": "#4e342e",
    "Beige": "#d7ccc8",        "Maroon": "#880e4f",
    "Olive": "#827717",        "Cream": "#fff8e1",
    "Silver": "#9e9e9e",       "Gold": "#f57f17",
    "Khaki": "#c8b856",        "Magenta": "#ad1457",
    "Teal": "#00695c",         "Off White": "#fafaf0",
    "Rust": "#bf360c",         "Turquoise": "#006064",
    "Charcoal": "#37474f",     "Multi": "#9c27b0",
    "Fluorescent Green": "#76ff03",
}

GENDER_ICON: dict[str, str] = {
    "Men": "♂",  "Women": "♀",
    "Boys": "👦", "Girls": "👧", "Unisex": "⚤",
}


# ══════════════════════════════════════════════════════════════
# CUSTOM CSS  ── premium dark-mode card UI
# ══════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* ── Header band ── */
    .header-band {
        background: linear-gradient(135deg, #0d0d1a 0%, #12192e 45%, #0f3460 100%);
        padding: 2.2rem 2rem 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.6rem;
        text-align: center;
        border: 1px solid #1e2a4a;
        box-shadow: 0 8px 32px rgba(233,69,96,0.15);
    }
    .header-band h1 {
        color: #e94560;
        font-size: 2.5rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-band .subtitle {
        color: #8892b0;
        font-size: 0.97rem;
        margin-top: 0.5rem;
        line-height: 1.7;
    }
    .header-band .subtitle em {
        color: #ccd6f6;
        font-style: normal;
        font-weight: 600;
    }

    /* ── Section title ── */
    .section-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: #e94560;
        border-left: 4px solid #e94560;
        padding: 0.35rem 0 0.35rem 0.8rem;
        margin: 2rem 0 1rem 0;
        background: linear-gradient(90deg, rgba(233,69,96,0.07) 0%, transparent 100%);
        border-radius: 0 6px 6px 0;
    }

    /* ── Product card ── */
    .product-card {
        background: #161b2e;
        border-radius: 12px;
        padding: 0.7rem 0.6rem 0.55rem 0.6rem;
        text-align: center;
        margin-bottom: 0.5rem;
        border: 1px solid #252d4a;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    }
    .product-card:hover {
        transform: translateY(-5px);
        border-color: #e94560;
        box-shadow: 0 8px 24px rgba(233,69,96,0.22);
    }
    .product-name {
        font-size: 0.76rem;
        font-weight: 600;
        color: #ccd6f6;
        margin: 0.4rem 0 0.3rem 0;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        min-height: 2.2em;
    }
    /* Badge row */
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        justify-content: center;
        margin-top: 0.4rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 3px;
        font-size: 0.61rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 20px;
        white-space: nowrap;
        letter-spacing: 0.3px;
    }
    .badge-type   { background: #e94560; color: #fff; }
    .badge-gender { background: #0f3460; color: #64b5f6; border: 1px solid #1565c0; }
    .badge-colour { color: #fff; border: 1px solid rgba(255,255,255,0.18); }
    .colour-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.35);
        flex-shrink: 0;
    }
    .usage-label {
        font-size: 0.59rem;
        color: #445;
        margin-top: 5px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ── Image placeholder ── */
    .img-placeholder {
        height: 168px;
        background: linear-gradient(135deg, #1e2a42, #252d4a);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #3d4f70;
        font-size: 2.8rem;
    }

    /* ── No-result box ── */
    .no-result {
        background: #161b2e;
        border: 2px dashed #e94560;
        border-radius: 12px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        color: #8892b0;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .no-result strong { color: #ccd6f6; }

    /* ── Stats strip ── */
    .stats-strip {
        background: #0d1117;
        border: 1px solid #1e2a4a;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.83rem;
        color: #8892b0;
        margin-bottom: 0.9rem;
    }
    .stats-strip strong { color: #e94560; }

    /* ── Search input override ── */
    div[data-testid="stTextInput"] > label { display: none !important; }
    div[data-testid="stTextInput"] > div > div > input {
        border-radius: 10px !important;
        border: 2px solid #1e2a4a !important;
        background: #0d1117 !important;
        color: #e6edf3 !important;
        padding: 0.7rem 1.1rem !important;
        font-size: 1rem !important;
    }
    div[data-testid="stTextInput"] > div > div > input:focus {
        border-color: #e94560 !important;
        box-shadow: 0 0 0 3px rgba(233,69,96,0.18) !important;
    }

    /* ── Landing hero ── */
    .landing-hero {
        text-align: center;
        margin-top: 3rem;
        padding: 3rem 2rem;
        background: linear-gradient(180deg, #0d1117 0%, #12192e 100%);
        border-radius: 16px;
        border: 1px solid #1e2a4a;
    }
    .landing-hero .hero-icon { font-size: 4.5rem; line-height: 1; }
    .landing-hero h3 { color: #ccd6f6; margin: 1rem 0 0.5rem 0; font-size: 1.3rem; }
    .landing-hero p  { color: #8892b0; font-size: 0.95rem; margin: 0; }
    .chip {
        display: inline-block;
        background: #1e2a4a;
        color: #64b5f6;
        padding: 4px 13px;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 4px 3px;
        border: 1px solid #2a3a5e;
        cursor: default;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .header-band h1 { font-size: 1.6rem; }
        .badge { font-size: 0.55rem; }
        .product-name { font-size: 0.7rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
# DATA LOADING  (cached once per session)
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="📦 Loading fashion dataset…")
def load_data() -> pd.DataFrame:
    """Load and clean styles.csv — identical to notebook."""
    df = pd.read_csv("styles.csv", on_bad_lines="skip")
    df = df.dropna(subset=["usage", "baseColour", "season", "productDisplayName"])
    df["year"] = df["year"].fillna(0).astype(int)
    df = df.reset_index(drop=True)
    return df


@st.cache_resource(show_spinner="🧠 Building similarity index…")
def build_tfidf(df: pd.DataFrame) -> tuple:
    """Build TF-IDF vectorizer and sparse matrix — compute similarities on-demand."""
    texts = df["articleType"] + " " + df["baseColour"] + " " + df["usage"]
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    matrix = tfidf.fit_transform(texts)
    return (tfidf, matrix)


# ══════════════════════════════════════════════════════════════
# ITEM MAP  (unchanged from notebook)
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# CATEGORY ORGANIZATION  ── organized by type
# ══════════════════════════════════════════════════════════════
CATEGORY_MAP: dict[str, list[str]] = {
    "👕 Topwear": [
        "Tshirts", "Shirts", "Tops", "Sweatshirts", 
        "Jackets", "Blazers", "Kurtas"
    ],
    "👖 Bottomwear": [
        "Jeans", "Trousers", "Shorts", "Track Pants",
        "Capris", "Leggings", "Skirts"
    ],
    "👗 Women Wear": [
        "Dresses", "Sarees", "Lehenga Choli", "Kurtas", "Tunics"
    ],
    "👟 Footwear": [
        "Casual Shoes", "Sports Shoes", "Sandals",
        "Flip Flops", "Heels", "Flats"
    ],
    "🎒 Accessories": [
        "Watches", "Sunglasses", "Handbags", "Belts",
        "Backpacks", "Wallets", "Jewellery"
    ],
}

# Flat ITEM_MAP for backward compatibility with search
ITEM_MAP: dict[str, list[str]] = {
    "tshirt":     ["Tshirts"],
    "tshirts":    ["Tshirts"],
    "shirt":      ["Shirts"],
    "shirts":     ["Shirts"],
    "top":        ["Tops"],
    "tops":       ["Tops"],
    "sweatshirt": ["Sweatshirts"],
    "sweatshirts": ["Sweatshirts"],
    "jacket":     ["Jackets"],
    "jackets":    ["Jackets"],
    "blazer":     ["Blazers"],
    "blazers":    ["Blazers"],
    "kurta":      ["Kurtas"],
    "kurtas":     ["Kurtas"],
    "pants":      ["Jeans", "Trousers", "Track Pants"],
    "jeans":      ["Jeans"],
    "trouser":    ["Trousers"],
    "trousers":   ["Trousers"],
    "shorts":     ["Shorts"],
    "short":      ["Shorts"],
    "capris":     ["Capris"],
    "leggings":   ["Leggings"],
    "skirt":      ["Skirts"],
    "skirts":     ["Skirts"],
    "dress":      ["Dresses"],
    "dresses":    ["Dresses"],
    "saree":      ["Sarees"],
    "sarees":     ["Sarees"],
    "lehenga":    ["Lehenga Choli"],
    "tunic":      ["Tunics"],
    "tunics":     ["Tunics"],
    "shoe":       ["Casual Shoes"],
    "shoes":      ["Casual Shoes"],
    "casual shoe": ["Casual Shoes"],
    "sports shoe": ["Sports Shoes"],
    "sandals":    ["Sandals"],
    "flip flops": ["Flip Flops"],
    "heels":      ["Heels"],
    "heel":       ["Heels"],
    "flats":      ["Flats"],
    "flat":       ["Flats"],
    "watch":      ["Watches"],
    "watches":    ["Watches"],
    "sunglasses": ["Sunglasses"],
    "handbag":    ["Handbags"],
    "handbags":   ["Handbags"],
    "belt":       ["Belts"],
    "belts":      ["Belts"],
    "backpack":   ["Backpacks"],
    "backpacks":  ["Backpacks"],
    "wallet":     ["Wallets"],
    "wallets":    ["Wallets"],
    "jewellery":  ["Jewellery"],
    "jewelry":    ["Jewellery"],
    "bag":        ["Handbags"],
    "bags":       ["Handbags"],
}

# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR CATEGORIES
# ══════════════════════════════════════════════════════════════
def get_subcategories_for_category(category: str, df: pd.DataFrame, categories_map: dict = None) -> list[str]:
    """
    Get available sub-categories from dataset for a given main category.
    Returns sub-categories that exist in both category map and dataset.
    """
    if categories_map is None:
        categories_map = CATEGORY_MAP
        
    if category == "All":
        return sorted(df["subCategory"].dropna().unique().tolist())
    
    mapped_subs = categories_map.get(category, [])
    available_subs = sorted(df["subCategory"].dropna().unique().tolist())
    # Match both exactly and case-insensitively
    result = []
    for sub in mapped_subs:
        # Exact match
        if sub in available_subs:
            result.append(sub)
        else:
            # Case-insensitive match
            for avail_sub in available_subs:
                if sub.lower() == avail_sub.lower():
                    result.append(avail_sub)
                    break
    return sorted(result)


def validate_categories(df: pd.DataFrame) -> dict:
    """
    Check which CATEGORY_MAP items exist in the dataset.
    Returns a report of found subcategories.
    """
    dataset_subs = set(df["subCategory"].dropna().unique())
    report = {}
    
    for cat_name, cat_subs in CATEGORY_MAP.items():
        found = [sub for sub in cat_subs if sub in dataset_subs]
        report[cat_name] = {
            "mapped": cat_subs,
            "found": found,
            "missing": [s for s in cat_subs if s not in dataset_subs]
        }
    
    return report


def build_dynamic_categories(df: pd.DataFrame) -> dict:
    """
    Build category mappings from actual dataset.
    Groups subcategories similar to predefined CATEGORY_MAP.
    Falls back to CATEGORY_MAP if perfect mapping exists.
    """
    dataset_subs = sorted(df["subCategory"].dropna().unique().tolist())
    
    # Check if all CATEGORY_MAP entries exist
    all_found = True
    for cat_subs in CATEGORY_MAP.values():
        for sub in cat_subs:
            if sub not in dataset_subs:
                all_found = False
                break
        if not all_found:
            break
    
    if all_found:
        # Use predefined mapping if all entries exist
        return CATEGORY_MAP
    
    # Otherwise, build from dataset
    # Group by common patterns
    dynamic_map = {}
    used_subs = set()
    
    for category_name, predefined_subs in CATEGORY_MAP.items():
        found_subs = [s for s in predefined_subs if s in dataset_subs]
        if found_subs:
            dynamic_map[category_name] = found_subs
            used_subs.update(found_subs)
    
    # Add any uncategorized subcategories to a general category
    uncategorized = [s for s in dataset_subs if s not in used_subs]
    if uncategorized:
        dynamic_map["📋 Other"] = uncategorized
    
    return dynamic_map


# ══════════════════════════════════════════════════════════════
#   Rules:
#   • Topwear → only Bottomwear (no random accessories)
#   • Bottomwear → only Topwear
#   • Footwear → compatible bottomwear/full-body
#   • Full-body (Dresses/Sarees) → footwear only
#   • Accessories → logical pairing target only
# ══════════════════════════════════════════════════════════════
OUTFIT_MAP: dict[str, list[str]] = {
    # ── Topwear ──────────────────────────────────────────────
    "Tshirts":      ["Jeans", "Shorts", "Track Pants"],
    "Shirts":       ["Jeans", "Trousers"],
    "Tops":         ["Jeans", "Trousers", "Skirts"],
    "Sweatshirts":  ["Jeans", "Track Pants"],
    "Jackets":      ["Jeans", "Trousers"],
    "Sweaters":     ["Jeans", "Trousers"],
    "Kurtas":       ["Trousers"],
    "Tunics":       ["Jeans", "Trousers"],

    # ── Bottomwear ───────────────────────────────────────────
    "Jeans":        ["Shirts", "Tshirts"],
    "Trousers":     ["Shirts", "Tshirts"],
    "Track Pants":  ["Tshirts", "Sweatshirts"],
    "Shorts":       ["Tshirts", "Shirts"],
    "Skirts":       ["Tops", "Shirts"],
    "Churidar":     ["Kurtas"],

    # ── Full-body ─────────────────────────────────────────────
    "Dresses":      ["Sandals", "Heels", "Flats"],
    "Sarees":       ["Heels", "Flats"],

    # ── Footwear ──────────────────────────────────────────────
    "Casual Shoes": ["Jeans", "Trousers", "Shorts"],
    "Sandals":      ["Dresses", "Skirts", "Shorts"],
    "Heels":        ["Dresses", "Trousers"],
    "Flats":        ["Dresses", "Skirts", "Trousers"],
    "Sports Shoes": ["Track Pants", "Shorts"],
    "Formal Shoes": ["Trousers"],

    # ── Accessories ───────────────────────────────────────────
    "Belts":        ["Jeans", "Trousers"],
    "Handbags":     ["Dresses", "Tops"],
    "Watches":      ["Shirts", "Tshirts"],
    "Sunglasses":   ["Tshirts", "Shirts", "Dresses"],
    "Ties":         ["Shirts"],
    "Caps":         ["Tshirts", "Shorts"],
    "Scarves":      ["Tops", "Shirts"],
}


# ══════════════════════════════════════════════════════════════
# CORE LOGIC  (unchanged from notebook)
# ══════════════════════════════════════════════════════════════
def normalize_input(user_input: str) -> str:
    """Spelling normalisation — same logic as notebook."""
    user_input = user_input.lower().strip()
    words = user_input.split()
    new_words = []
    for w in words:
        if w in ("pents", "pant"):
            new_words.append("pants")
        elif w == "watchs":
            new_words.append("watches")
        elif w in ("t-shirt", "tshirt", "tshirts", "t"):
            new_words.append("tshirts")
        else:
            new_words.append(w)
    return " ".join(new_words)


def parse_input(user_input: str, df: pd.DataFrame) -> tuple:
    """
    Returns (color, items_list, gender, category)
    Enhanced to detect categories from input.
    """
    user_input = normalize_input(user_input)
    words = user_input.split()

    color = None
    items: list[str] = []
    gender = None
    detected_category = None

    # Category detection (check category names and keywords)
    for category_name, sub_list in CATEGORY_MAP.items():
        clean_cat = category_name.lower().strip()
        # Check if any sub-category name is in input
        for sub in sub_list:
            if sub.lower() in user_input:
                detected_category = category_name
                break
        # Check category keywords
        if not detected_category:
            keywords = {
                "topwear": "👕 Topwear",
                "top wear": "👕 Topwear",
                "tshirt": "👕 Topwear",
                "shirt": "👕 Topwear",
                "jacket": "👕 Topwear",
                "sweatshirt": "👕 Topwear",
                "bottomwear": "👖 Bottomwear",
                "bottom wear": "👖 Bottomwear",
                "jeans": "👖 Bottomwear",
                "trousers": "👖 Bottomwear",
                "pants": "👖 Bottomwear",
                "shorts": "👖 Bottomwear",
                "women wear": "👗 Women Wear",
                "dress": "👗 Women Wear",
                "saree": "👗 Women Wear",
                "lehenga": "👗 Women Wear",
                "footwear": "👟 Footwear",
                "shoes": "👟 Footwear",
                "sandals": "👟 Footwear",
                "heels": "👟 Footwear",
                "flats": "👟 Footwear",
                "accessories": "🎒 Accessories",
                "accessory": "🎒 Accessories",
                "watch": "🎒 Accessories",
                "sunglasses": "🎒 Accessories",
                "handbag": "🎒 Accessories",
                "belt": "🎒 Accessories",
            }
            for keyword, cat in keywords.items():
                if keyword in user_input:
                    detected_category = cat
                    break
        if detected_category:
            break

    # Colour detection
    for c in df["baseColour"].dropna().unique():
        if c.lower() in user_input:
            color = c
            break

    # Item detection via ITEM_MAP
    for w in words:
        if w in ITEM_MAP:
            items.extend(ITEM_MAP[w])

    # Fallback: direct match against dataset article types
    if not items:
        for art in df["articleType"].unique():
            if art.lower() in user_input:
                items.append(art)

    # Gender detection (women BEFORE men to prevent "women" → "men" match)
    if "women" in user_input or "woman" in user_input:
        gender = "Women"
    elif "men" in user_input or "man" in user_input:
        gender = "Men"
    elif "girl" in user_input:
        gender = "Girls"
    elif "boy" in user_input:
        gender = "Boys"

    return color, list(set(items)), gender, detected_category


def smart_recommend_full(
    user_input: str,
    df: pd.DataFrame,
    top_n: int = 10,
    gender_override: str = "All",
    category_override: str = "All",
    categories_map: dict = None,
) -> pd.DataFrame | None:
    """
    Main recommendation engine — enhanced with category detection.
    UI overrides (gender, category, subCategory) applied on top.
    Priority: UI selections > Query detection > Items from ITEM_MAP
    """
    if categories_map is None:
        categories_map = CATEGORY_MAP
        
    color, items, gender, detected_category = parse_input(user_input, df)

    # ── GENDER FILTER ───────────────────────────────────────
    if gender_override != "All":
        gender = gender_override
    
    data = df.copy()
    if gender:
        data = data[data["gender"] == gender]

    # ── CATEGORY FILTER ────────────────────────────────────
    # Priority: UI category_override > detected category from query
    active_category = None
    if category_override != "All":
        active_category = category_override
    elif detected_category and detected_category != "All":
        active_category = detected_category
    
    if active_category:
        category_subs = categories_map.get(active_category, [])
        if category_subs:
            data = data[data["subCategory"].isin(category_subs)]

    # ── ITEM TYPE FILTER ───────────────────────────────────
    if items:
        data = data[data["articleType"].isin(items)]

    # ── COLOUR FILTER ──────────────────────────────────────
    if color:
        data = data[data["baseColour"] == color]

    if data.empty:
        return None

    return data.head(top_n)


# ══════════════════════════════════════════════════════════════
# [FIX #1 + #2 + #5]  OUTFIT RECOMMENDATIONS  ── context-aware
#   Changes vs v1:
#   ✅ Gender is mandatory – same gender as main results
#   ✅ Usage match (Casual / Formal / Sports …)
#   ✅ Colour preference (soft: prefer match, graceful fallback)
#   ✅ No random .sample() – sorted, deterministic selection
#   ✅ Excludes original searched article types
# ══════════════════════════════════════════════════════════════
def get_outfit_recommendations(
    search_results: pd.DataFrame,
    df: pd.DataFrame,
    gender: str | None,
    color: str | None,
    usage: str | None,
    top_n: int = 10,
) -> pd.DataFrame | None:
    """
    Suggests complementary items filtered to the same context
    (gender, usage, preferred colour) as the main search results.

    Parameters
    ----------
    search_results : result DataFrame from smart_recommend_full
    df             : full cleaned dataset
    gender         : resolved gender (Men / Women / Boys / Girls / None)
    color          : resolved colour string from parse_input
    usage          : dominant usage in search_results
    top_n          : maximum items to return
    """
    # Step 1 – collect target article types from OUTFIT_MAP
    found_types = search_results["articleType"].unique().tolist()
    outfit_types: list[str] = []
    for art in found_types:
        outfit_types.extend(OUTFIT_MAP.get(art, []))

    # Remove duplicates and the originally searched types
    outfit_types = list(dict.fromkeys(t for t in outfit_types if t not in found_types))
    if not outfit_types:
        return None

    # Step 2 – pool of candidates
    pool = df[df["articleType"].isin(outfit_types)].copy()
    if pool.empty:
        return None

    # [FIX #2] CRITICAL: gender must match
    if gender:
        gender_pool = pool[pool["gender"] == gender]
        pool = gender_pool if not gender_pool.empty else pool

    # Usage consistency (Casual / Formal / Sports …)
    if usage:
        usage_pool = pool[pool["usage"] == usage]
        pool = usage_pool if not usage_pool.empty else pool

    if pool.empty:
        return None

    # Step 3 – colour preference (soft)
    # Prefer colour-matched items; pad remaining slots with any from pool
    if color:
        colour_pool = pool[pool["baseColour"] == color]
        if not colour_pool.empty:
            selected = colour_pool.head(top_n)
            if len(selected) < top_n:
                rest = pool[~pool.index.isin(selected.index)].head(top_n - len(selected))
                selected = pd.concat([selected, rest])
            return selected.head(top_n)

    # Step 4 – no colour constraint: return deterministic top results
    return pool.head(top_n)


# ══════════════════════════════════════════════════════════════
# [FIX #2]  SIMILAR ITEMS  ── gender-consistent
# ══════════════════════════════════════════════════════════════
def get_similar_items(
    index: int,
    tfidf_data: tuple,
    df: pd.DataFrame,
    gender: str | None,
    top_n: int = 5,
) -> pd.DataFrame | None:
    """
    TF-IDF cosine similarity – returns items of the SAME gender.
    Computes similarity on-demand for memory efficiency.
    """
    tfidf, matrix = tfidf_data
    
    if index >= len(df):
        return None

    # Compute similarity only for this specific item vs all others
    query_vector = matrix[index]
    sim_scores_array = cosine_similarity(query_vector, matrix).flatten()
    
    sim_scores = sorted(
        enumerate(sim_scores_array), key=lambda x: x[1], reverse=True
    )
    valid_scores = [s for s in sim_scores[1:] if s[0] < len(df)]
    if not valid_scores:
        return None

    similar_df = df.iloc[[i[0] for i in valid_scores]].copy()

    # [FIX #2] Filter to same gender; fall back only if nothing found
    if gender:
        gender_similar = similar_df[similar_df["gender"] == gender]
        similar_df = gender_similar if not gender_similar.empty else similar_df

    return similar_df.head(top_n)


# ══════════════════════════════════════════════════════════════
# IMAGE HELPER
# ══════════════════════════════════════════════════════════════
def load_image(product_id, size: tuple = (220, 280)) -> Image.Image | None:
    """Load a product image; returns PIL Image or None."""
    path = os.path.join("images", f"{product_id}.jpg")
    if os.path.exists(path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(size, Image.LANCZOS)
            return img
        except Exception:
            return None
    return None


# ══════════════════════════════════════════════════════════════
# [UI] PRODUCT CARD  ── gender badge + colour dot (NEW in v2)
# ══════════════════════════════════════════════════════════════
def render_product_card(col, row: pd.Series) -> None:
    """Render a single product card with image + badge row."""
    img = load_image(row["id"])

    colour_name = str(row.get("baseColour", ""))
    colour_hex  = COLOUR_HEX.get(colour_name, "#3a3f5c")
    gender_val  = str(row.get("gender", ""))
    gender_icon = GENDER_ICON.get(gender_val, "")
    usage_val   = str(row.get("usage", ""))
    art_type    = str(row.get("articleType", ""))
    prod_name   = str(row.get("productDisplayName", ""))

    with col:
        # Image or placeholder
        if img:
            st.image(img, use_container_width=True)
        else:
            st.markdown(
                '<div class="img-placeholder">👗</div>',
                unsafe_allow_html=True,
            )

        # Card HTML
        st.markdown(
            f"""
            <div class="product-card">
                <div class="product-name" title="{prod_name}">{prod_name}</div>
                <div class="badge-row">
                    <span class="badge badge-type">{art_type}</span>
                    <span class="badge badge-gender">{gender_icon} {gender_val}</span>
                    <span class="badge badge-colour"
                          style="background:{colour_hex};">
                        <span class="colour-dot"
                              style="background:{colour_hex};
                                     border:1px solid rgba(255,255,255,0.4);">
                        </span>
                        {colour_name}
                    </span>
                </div>
                <div class="usage-label">{usage_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_product_grid(results: pd.DataFrame, cols_per_row: int = 5) -> None:
    """Render a responsive grid of product cards."""
    chunks = [results.iloc[i: i + cols_per_row] for i in range(0, len(results), cols_per_row)]
    for chunk in chunks:
        cols = st.columns(cols_per_row)
        for col, (_, product) in zip(cols, chunk.iterrows()):
            render_product_card(col, product)


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
def main() -> None:

    # ── Header ─────────────────────────────────────────────
    st.markdown(
        """
        <div class="header-band">
            <h1>👕 AI Fashion Recommender System</h1>
            <div class="subtitle">
                Search outfits using natural language &nbsp;·&nbsp;
                Try: <em>"black shirt for men"</em> &nbsp;·&nbsp;
                <em>"white tshirts for women"</em> &nbsp;·&nbsp;
                <em>"blue jeans for boys"</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data (cached) ──────────────────────────────────
    df = load_data()
    tfidf_data = build_tfidf(df)
    
    # ── Build dynamic categories from dataset ────────────────
    dynamic_categories = build_dynamic_categories(df)

    # ── Category Info (collapsible) ──────────────────────────
    with st.expander("📂 Browse by Category", expanded=False):
        category_info = "**Search by category or sub-category:**\n\n"
        for cat, subs in dynamic_categories.items():
            category_info += f"**{cat}**  \n"
            category_info += ", ".join(subs) + "\n\n"
        st.markdown(category_info)

    # ── Search row ──────────────────────────────────────────
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            "query",
            placeholder=(
                'e.g.  "black shirt for men"  ·  "red dress for women"  ·  "blue jeans"'
            ),
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")

    # ── [FIX #3] Filters  ───────────────────────────────────
    st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)

    with f1:
        # Gender values sourced from the actual dataset – no hardcoding
        gender_vals = sorted(df["gender"].dropna().unique().tolist())
        gender_filter = st.selectbox("👤 Gender", ["All"] + gender_vals, index=0)

    with f2:
        # Main category selector - using dynamic categories
        category_filter = st.selectbox(
            "🏷️ Category",
            ["All"] + list(dynamic_categories.keys()),
            index=0
        )

    with f3:
        top_n = st.slider("📋 Results", min_value=5, max_value=20, value=10, step=5)

    st.divider()

    # ── Run search ──────────────────────────────────────────
    if search_clicked or user_query:
        if not user_query.strip():
            st.warning("⚠️ Please type something in the search box first.")
            return

        with st.spinner("✨ Searching best outfits…"):
            results = smart_recommend_full(
                user_query,
                df,
                top_n=top_n,
                gender_override=gender_filter,
                category_override=category_filter,
                categories_map=CATEGORY_MAP,
            )

        # ── Resolve context shared by all three sections ────
        parsed_color, parsed_items, parsed_gender, parsed_category = parse_input(user_query, df)
        # UI override wins (mirrors smart_recommend_full logic)
        effective_gender = gender_filter if gender_filter != "All" else parsed_gender
        effective_color  = parsed_color

        # ── ✅  TOP RESULTS ─────────────────────────────────
        st.markdown(
            '<div class="section-title">✅ Top Results</div>',
            unsafe_allow_html=True,
        )

        if results is None or results.empty:
            st.markdown(
                """
                <div class="no-result">
                    😕 <strong>No matching fashion items found.</strong><br><br>
                    Try:
                    <em>"black jeans for men"</em> &nbsp;·&nbsp;
                    <em>"red dress for women"</em> &nbsp;·&nbsp;
                    <em>"white tshirt"</em>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        # Stats strip
        gender_label = effective_gender or "All genders"
        color_label  = effective_color  or "Any colour"
        st.markdown(
            f"""
            <div class="stats-strip">
                🔎 &nbsp;<strong>{len(results)}</strong> result(s) &nbsp;·&nbsp;
                Query: <strong>{user_query}</strong> &nbsp;·&nbsp;
                Gender: <strong>{gender_label}</strong> &nbsp;·&nbsp;
                Colour: <strong>{color_label}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_product_grid(results, cols_per_row=5)

        # ── 👗  OUTFIT SUGGESTIONS ──────────────────────────
        # Dominant usage in results (Casual / Formal / Sports …)
        usage_series = results["usage"].dropna()
        dominant_usage = usage_series.mode()[0] if not usage_series.empty else None

        outfit_results = get_outfit_recommendations(
            search_results=results,
            df=df,
            gender=effective_gender,
            color=effective_color,
            usage=dominant_usage,
            top_n=top_n,
        )

        st.markdown(
            '<div class="section-title">👗 Outfit Suggestions</div>',
            unsafe_allow_html=True,
        )
        if outfit_results is not None and not outfit_results.empty:
            render_product_grid(outfit_results, cols_per_row=5)
        else:
            st.markdown(
                '<div class="no-result" style="padding:1rem;">'
                "No outfit suggestions available for this combination.</div>",
                unsafe_allow_html=True,
            )

        # ── 🔁  SIMILAR ITEMS ───────────────────────────────
        first_idx = results.index[0]
        similar = get_similar_items(
            index=first_idx,
            tfidf_data=tfidf_data,
            df=df,
            gender=effective_gender,
            top_n=top_n,
        )

        if similar is not None and not similar.empty:
            st.markdown(
                '<div class="section-title">🔁 Similar Items</div>',
                unsafe_allow_html=True,
            )
            render_product_grid(similar, cols_per_row=5)

    else:
        # ── Landing hero ────────────────────────────────────
        st.markdown(
            """
            <div class="landing-hero">
                <div class="hero-icon">👗</div>
                <h3>Discover Your Perfect Outfit</h3>
                <p>Search using natural language — colour, type, and gender all work!</p>
                <br>
                <span class="chip">black shirt for men</span>
                <span class="chip">red dress for women</span>
                <span class="chip">blue jeans</span>
                <span class="chip">white tshirts for boys</span>
                <span class="chip">casual shoes for women</span>
                <span class="chip">grey trousers for men</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()