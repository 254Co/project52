# File: chen3/operations/dashboard.py
"""
Streamlit dashboard for Chen3 package.
"""
import numpy as np
import pandas as pd
import streamlit as st

from chen3 import (
    ChenModel,
    EquityParams,
    ModelParams,
    MonteCarloPricer,
    RateParams,
    Settings,
    make_simulator,
)
from chen3.payoffs import Vanilla


@st.cache_resource
def initialize_engine(model_params, settings_params):
    rate = RateParams(**model_params["rate"])
    equity = EquityParams(**model_params["equity"])
    corr = np.eye(3)
    model = ChenModel(ModelParams(rate, equity, corr))
    settings = Settings(**settings_params)
    sim = make_simulator(model, settings)
    return sim, settings, rate


def run_dashboard():
    st.title("Chen3 Risk Dashboard")
    st.sidebar.header("Model Parameters")
    r0 = st.sidebar.number_input("Initial rate (r0)", value=0.03)
    theta = st.sidebar.number_input("Rate theta", value=0.03)
    sigma_r = st.sidebar.number_input("Rate sigma", value=0.01)
    kappa = st.sidebar.number_input("Rate kappa", value=0.1)
    mu = st.sidebar.number_input("Equity drift (mu)", value=0.05)
    q = st.sidebar.number_input("Dividend yield (q)", value=0.02)
    S0 = st.sidebar.number_input("Initial spot (S0)", value=100.0)
    v0 = st.sidebar.number_input("Initial var (v0)", value=0.04)
    kappa_v = st.sidebar.number_input("Vol kappa", value=1.5)
    theta_v = st.sidebar.number_input("Vol theta", value=0.04)
    sigma_v = st.sidebar.number_input("Vol sigma", value=0.3)
    n_paths = st.sidebar.number_input("Paths", value=10000)
    n_steps = st.sidebar.number_input("Steps", value=252)

    model_params = {
        "rate": {"kappa": kappa, "theta": theta, "sigma": sigma_r, "r0": r0},
        "equity": {
            "mu": mu,
            "q": q,
            "S0": S0,
            "v0": v0,
            "kappa_v": kappa_v,
            "theta_v": theta_v,
            "sigma_v": sigma_v,
        },
    }
    settings_params = {
        "seed": 42,
        "backend": "cpu",
        "n_paths": n_paths,
        "n_steps": n_steps,
        "dt": 1 / 252,
    }
    sim, settings, rate = initialize_engine(model_params, settings_params)

    if st.sidebar.button("Run Monte Carlo"):
        paths = sim.generate()
        payoff = Vanilla(strike=st.sidebar.number_input("Strike", value=100.0))
        pricer = MonteCarloPricer(
            payoff,
            discount_curve=lambda T: np.exp(-rate.theta * T),
            dt=settings.dt,
            n_steps=settings.n_steps,
        )
        price = pricer.price(paths)
        st.metric("Monte Carlo Price", f"{price:.4f}")
