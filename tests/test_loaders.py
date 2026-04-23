import numpy as np
from src.data_io import load_324_ratings, load_movie_meta, train_test_split_holdout

def test_324_ratings_shape():
    df = load_324_ratings()
    assert len(df) == 324
    # Real modalities in data — not the plan's assumed set
    assert set(df['modality'].unique()) == {'all', 'metadata', 'poster', 'synopsis'}
    # must have year after join
    assert 'year' in df.columns
    assert df['year'].notna().sum() > 300  # most should resolve

def test_movie_meta_is_dict_by_tmdb_id():
    meta = load_movie_meta()
    assert isinstance(meta, dict)
    assert len(meta) >= 80
    # each entry has year
    some_key = next(iter(meta))
    assert 'year' in meta[some_key]

def test_holdout_split_disjoint():
    df = load_324_ratings()
    train, test = train_test_split_holdout(df, test_frac=0.2, seed=42)
    assert set(train.index).isdisjoint(set(test.index))
    assert len(train) + len(test) == 324

def test_holdout_split_reproducible():
    df = load_324_ratings()
    t1, _ = train_test_split_holdout(df, 0.2, 42)
    t2, _ = train_test_split_holdout(df, 0.2, 42)
    assert list(t1.index) == list(t2.index)
