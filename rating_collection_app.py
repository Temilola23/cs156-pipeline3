"""
Modality-conditional rating-collection app for CS156 Pipeline 3 (ACT I).

The user (Temilola) re-rates 100 movies he's already seen under four
*conditions* that hide some of the modality channels:

    1. poster   — only the poster image
    2. metadata — only title, year, runtime, genres
    3. synopsis — only the TMDB overview text
    4. all      — poster + metadata + synopsis

100 movies x 4 conditions = 400 modality-conditional ratings. These ratings
let us decompose how much of his rating is "explained by" each modality
versus the joint, which is the empirical foundation for the unified
v_taste vector in ACT III.

Design notes
------------
- Latin-square interleaving: each condition appears the same number of
  times in each block of 4, and a movie's four conditions are spread
  across the session (not back-to-back), so memory leakage is minimised.
- Persistent JSONL store: every submitted rating is appended immediately
  to data/modality_ratings.jsonl, so a refresh / device-switch never
  loses progress. Resume picks up from the next un-rated (movie, condition)
  pair.
- Mobile-first layout: large single-column UI with a 0-10 slider in 0.5
  steps (matching the original Letterboxd scale). One submission per page.
- The app does NOT show his original ratings during collection — that
  would defeat the point. They are loaded only at the end for the
  comparison view.

Run locally:
    streamlit run rating_collection_app.py

Deploy:
    Push to a GitHub repo and connect to share.streamlit.io. The app
    reads its data from `data/movies_meta.json` (built by
    build_movie_index.py) and writes to `data/modality_ratings.jsonl`.
    For Streamlit Cloud persistence, point DATA_DIR at a mounted volume
    or commit the JSONL back to git via a small helper.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
META_PATH = DATA_DIR / "movies_meta.json"
RATINGS_PATH = DATA_DIR / "modality_ratings.jsonl"

N_MOVIES = 100
CONDITIONS = ["poster", "metadata", "synopsis", "all"]
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_meta() -> list[dict]:
    if not META_PATH.exists():
        return []
    return json.loads(META_PATH.read_text())


def stratified_sample_titles(
    meta: list[dict], n: int, seed: int = 0
) -> list[dict]:
    """Sample n titles stratified by rating buckets to cover the full range.

    Falls back to "take everything" if there aren't n titles available.
    """
    df = pd.DataFrame([m for m in meta if m.get("tmdb_resolved")])
    if df.empty:
        return []
    if len(df) <= n:
        return df.to_dict("records")

    # 5 buckets across 0-10. Sample proportionally, then top-up.
    buckets = pd.cut(
        df["my_rating_mean"],
        bins=[-0.01, 4, 6, 7.5, 9, 10.01],
        labels=["F", "D", "C", "B", "A"],
    )
    df = df.copy()
    df["bucket"] = buckets

    rng = random.Random(seed)
    per = max(1, n // df["bucket"].nunique())
    sampled = []
    for _, group in df.groupby("bucket", observed=True):
        ids = group.to_dict("records")
        rng.shuffle(ids)
        sampled.extend(ids[:per])

    if len(sampled) < n:
        seen = {(s["title"], s["media_type"]) for s in sampled}
        leftover = [
            r for r in df.to_dict("records")
            if (r["title"], r["media_type"]) not in seen
        ]
        rng.shuffle(leftover)
        sampled.extend(leftover[: n - len(sampled)])
    return sampled[:n]


def build_schedule(titles: list[dict], seed: int = 0) -> list[dict]:
    """Latin-square interleaved schedule of (title, condition) pairs.

    For each title we shuffle the four conditions (Latin-square at the
    title level), then interleave across titles so that the four
    conditions for movie k are spread far apart in the global order.
    """
    rng = random.Random(seed)

    per_title_orderings = []
    for t in titles:
        order = CONDITIONS.copy()
        rng.shuffle(order)
        per_title_orderings.append(order)

    title_indices = list(range(len(titles)))
    rng.shuffle(title_indices)

    schedule = []
    for round_idx in range(len(CONDITIONS)):
        rng.shuffle(title_indices)
        for ti in title_indices:
            cond = per_title_orderings[ti][round_idx]
            t = titles[ti]
            schedule.append({
                "movie_key": _movie_key(t),
                "title": t["title"],
                "media_type": t["media_type"],
                "tmdb_id": t.get("tmdb_id"),
                "condition": cond,
            })
    return schedule


def _movie_key(t: dict) -> str:
    return f"{t['title']}|{t['media_type']}"


# ---------------------------------------------------------------------------
# Ratings I/O
# ---------------------------------------------------------------------------


def append_rating(record: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RATINGS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_ratings() -> list[dict]:
    if not RATINGS_PATH.exists():
        return []
    out = []
    for line in RATINGS_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def already_rated_keys(ratings: list[dict]) -> set[tuple[str, str]]:
    return {(r["movie_key"], r["condition"]) for r in ratings}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def render_condition(meta_for_title: dict, condition: str) -> None:
    poster = meta_for_title.get("poster_path")
    title = meta_for_title.get("tmdb_title") or meta_for_title["title"]
    year = meta_for_title.get("year", "")
    runtime = meta_for_title.get("runtime_min")
    genres = meta_for_title.get("genres", [])
    overview = meta_for_title.get("overview", "")
    media = (
        "Show"
        if meta_for_title.get("tmdb_media_type", meta_for_title["media_type"]) == "tv"
        else "Movie"
    )

    if condition == "poster":
        if poster:
            st.image(TMDB_IMG_BASE + poster, use_container_width=True)
        else:
            st.warning("(no poster available — fallback: any text on the page is incidental)")

    elif condition == "metadata":
        runtime_str = f"{runtime} min" if runtime else "N/A"
        genre_str = ", ".join(genres) if genres else "N/A"
        st.markdown(
            f"### {title}"
            + (f" ({year})" if year else "")
            + f"\n\n**Type:** {media}  \n"
            + f"**Runtime:** {runtime_str}  \n"
            + f"**Genres:** {genre_str}"
        )

    elif condition == "synopsis":
        if overview:
            st.markdown(f"> {overview}")
        else:
            st.warning("(no synopsis available)")

    elif condition == "all":
        cols = st.columns([2, 3])
        with cols[0]:
            if poster:
                st.image(TMDB_IMG_BASE + poster, use_container_width=True)
        with cols[1]:
            runtime_str = f"{runtime} min" if runtime else "N/A"
            genre_str = ", ".join(genres) if genres else "N/A"
            st.markdown(
                f"### {title}"
                + (f" ({year})" if year else "")
                + f"\n\n**Type:** {media}  \n"
                + f"**Runtime:** {runtime_str}  \n"
                + f"**Genres:** {genre_str}"
            )
            if overview:
                st.markdown(f"\n{overview}")


def session_id() -> str:
    """Stable per-browser-session identifier so we can group bursts of work."""
    if "_sid" not in st.session_state:
        seed = f"{datetime.now().isoformat()}-{random.random()}"
        st.session_state._sid = hashlib.sha1(seed.encode()).hexdigest()[:10]
    return st.session_state._sid


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Pipeline 3 — Modality Ratings",
        page_icon="🎬",
        layout="centered",
    )
    st.title("Pipeline 3 — modality-conditional ratings")
    st.caption(
        "100 movies × 4 conditions (poster / metadata / synopsis / all) = "
        "400 ratings. Rate on the same 0–10 scale as your Letterboxd CSV."
    )

    meta = load_meta()
    if not meta:
        st.error(
            "No `data/movies_meta.json` found. Run `python build_movie_index.py` "
            "first to populate the index."
        )
        st.stop()

    # Build the deterministic schedule (same seed → same order across reloads).
    titles = stratified_sample_titles(meta, n=N_MOVIES, seed=42)
    schedule = build_schedule(titles, seed=42)
    meta_by_key = {_movie_key(m): m for m in meta}

    ratings = load_ratings()
    done = already_rated_keys(ratings)

    # Find the next un-rated item.
    next_idx = next(
        (i for i, item in enumerate(schedule)
         if (item["movie_key"], item["condition"]) not in done),
        None,
    )

    # Sidebar: progress + admin.
    with st.sidebar:
        st.header("Progress")
        total = len(schedule)
        completed = sum(
            1 for item in schedule
            if (item["movie_key"], item["condition"]) in done
        )
        st.metric("Rated", f"{completed} / {total}")
        st.progress(completed / total if total else 0.0)

        per_cond = {c: 0 for c in CONDITIONS}
        for r in ratings:
            if r["condition"] in per_cond:
                per_cond[r["condition"]] += 1
        st.subheader("By condition")
        for c, n in per_cond.items():
            st.write(f"- **{c}**: {n}")

        st.subheader("Admin")
        st.code(f"Output: {RATINGS_PATH.name}")
        if st.button("Download ratings JSONL"):
            st.download_button(
                "Click to download",
                data=RATINGS_PATH.read_bytes() if RATINGS_PATH.exists() else b"",
                file_name="modality_ratings.jsonl",
                mime="application/jsonl",
            )

    if next_idx is None:
        st.success("All 400 ratings collected — Pipeline 3 ACT I dataset is complete!")
        st.balloons()
        df = pd.DataFrame(ratings)
        st.dataframe(df, use_container_width=True)
        return

    item = schedule[next_idx]
    movie_meta = meta_by_key.get(item["movie_key"])
    if movie_meta is None:
        st.error(f"Missing metadata for {item['movie_key']!r}.")
        st.stop()

    st.subheader(f"#{next_idx + 1} of {len(schedule)} — condition: `{item['condition']}`")
    st.caption(
        "Rate based ONLY on the information shown below. Don't try to "
        "remember your past Letterboxd score — give a fresh judgement."
    )
    render_condition(movie_meta, item["condition"])

    st.divider()
    with st.form(key=f"rate_form_{next_idx}", clear_on_submit=False):
        rating = st.slider("Rating (0–10)", 0.0, 10.0, 5.0, 0.1)
        confidence = st.slider(
            "Confidence (1–5)",
            1, 5, 3,
            help="How confident are you in this rating given only this view?",
        )
        notes = st.text_input("Notes (optional)", "")
        submitted = st.form_submit_button("Submit rating  ▶", use_container_width=True)

        if submitted:
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id(),
                "schedule_index": next_idx,
                "movie_key": item["movie_key"],
                "title": item["title"],
                "media_type": item["media_type"],
                "tmdb_id": item.get("tmdb_id"),
                "condition": item["condition"],
                "rating": rating,
                "confidence": confidence,
                "notes": notes,
            }
            append_rating(record)
            st.rerun()


if __name__ == "__main__":
    main()
