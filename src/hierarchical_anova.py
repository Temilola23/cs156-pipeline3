"""
Bayesian hierarchical one-way ANOVA (PyMC) for modality effect on ratings.

Model:
  μ ~ Normal(3.5, 1)            # grand mean
  σ_modality ~ HalfNormal(1)     # between-modality sd
  a_m_raw ~ Normal(0, 1)         # non-centered modality offset (m ∈ {0,1,2,3})
  a_m = σ_modality * a_m_raw
  σ_movie ~ HalfNormal(1)        # between-movie sd
  b_i_raw ~ Normal(0, 1)         # non-centered movie effect
  b_i = σ_movie * b_i_raw
  σ_obs ~ HalfNormal(1)          # observation noise
  y_ij ~ Normal(μ + a_m[j] + b_i[j], σ_obs)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


def build_hierarchical_anova_model(
    df: pd.DataFrame,
) -> tuple[pm.Model, dict]:
    """
    Build and return a PyMC Bayesian hierarchical ANOVA model.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: tmdb_id, rating, modality
        - rating: float, observed ratings (1-5 scale)
        - modality: str, one of {'all', 'metadata', 'poster', 'synopsis'}
        - tmdb_id: int or str, movie identifier

    Returns
    -------
    model : pm.Model
        Compiled PyMC model (ready for sampling or prior draws)
    data_dict : dict
        Encoding dict with keys:
        - 'modality_idx': (n,) int array, encoded modality indices
        - 'movie_idx': (n,) int array, encoded movie indices
        - 'ratings': (n,) float array, observed ratings
        - 'modality_map': dict mapping modality string to int
        - 'movie_map': dict mapping tmdb_id to int
    """
    # Encode modality
    modality_map = {'all': 0, 'metadata': 1, 'poster': 2, 'synopsis': 3}
    modality_idx = np.array([modality_map[m] for m in df['modality'].values], dtype=int)
    n_modalities = len(modality_map)

    # Encode movie (tmdb_id → integer)
    unique_movies = sorted(df['tmdb_id'].unique())
    movie_map = {tmdb_id: i for i, tmdb_id in enumerate(unique_movies)}
    movie_idx = np.array([movie_map[int(tid)] for tid in df['tmdb_id'].values], dtype=int)
    n_movies = len(unique_movies)

    # Extract ratings
    ratings = df['rating'].values.astype(np.float32)

    print(f"[ANOVA] Data: {len(df)} ratings, {n_modalities} modalities, {n_movies} movies")

    # Build model
    with pm.Model() as model:
        # Priors
        # Rating scale is 1-5, so grand mean around 3.5 ± 1.5 is reasonable
        mu = pm.Normal('mu', mu=3.5, sigma=1.5)

        # Variance priors: weakly informative
        # For a 1-5 scale, max SD should be ~2, so HalfNormal(0.5) is reasonable
        sigma_modality = pm.HalfNormal('sigma_modality', sigma=0.5)
        sigma_movie = pm.HalfNormal('sigma_movie', sigma=0.5)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.5)

        # Non-centered parameterization for modality offset
        a_m_raw = pm.Normal('a_m_raw', mu=0, sigma=1.0, shape=n_modalities)
        a_m = pm.Deterministic('a_m', sigma_modality * a_m_raw)

        # Non-centered parameterization for movie effect
        b_i_raw = pm.Normal('b_i_raw', mu=0, sigma=1.0, shape=n_movies)
        b_i = pm.Deterministic('b_i', sigma_movie * b_i_raw)

        # Likelihood
        mu_ij = mu + a_m[modality_idx] + b_i[movie_idx]
        y = pm.Normal('y', mu=mu_ij, sigma=sigma_obs, observed=ratings)

    data_dict = {
        'modality_idx': modality_idx,
        'movie_idx': movie_idx,
        'ratings': ratings,
        'modality_map': modality_map,
        'movie_map': movie_map,
    }

    return model, data_dict


def fit_hierarchical_anova(
    df: pd.DataFrame,
    n_tune: int = 1000,
    n_draw: int = 1000,
    n_chains: int = 2,
    seed: int = 42,
    target_accept: float = 0.95,
) -> az.InferenceData:
    """
    Fit the hierarchical ANOVA model and return posterior samples.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: tmdb_id, rating, modality
    n_tune : int
        Number of tuning steps per chain (default 1000)
    n_draw : int
        Number of post-tuning draws per chain (default 1000)
    n_chains : int
        Number of chains (default 2; 1 acceptable if slow)
    seed : int
        Random seed (default 42)
    target_accept : float
        Target acceptance rate for NUTS (default 0.95)

    Returns
    -------
    idata : az.InferenceData
        ArviZ InferenceData with posterior samples, diagnostics, etc.
    """
    model, data_dict = build_hierarchical_anova_model(df)

    print(f"[ANOVA] Sampling: {n_chains} chains × ({n_tune} tune + {n_draw} draw) ...")
    with model:
        idata = pm.sample(
            draws=n_draw,
            tune=n_tune,
            chains=n_chains,
            random_seed=seed,
            target_accept=target_accept,
            progressbar=True,
            return_inferencedata=True,
        )

    return idata
