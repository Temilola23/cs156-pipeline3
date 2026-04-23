import numpy as np
from src.kalman import KalmanFilter, RTSSmoother, UKF

def test_kalman_converges_on_constant():
    np.random.seed(0)
    kf = KalmanFilter(F=np.eye(1), H=np.eye(1), Q=np.array([[0.01]]), R=np.array([[0.5]]))
    kf.init(np.zeros(1), np.eye(1))
    xs = []
    for y in np.random.normal(5.0, np.sqrt(0.5), 200):
        kf.predict(); kf.update(np.array([y])); xs.append(kf.x.copy())
    assert abs(xs[-1][0] - 5.0) < 0.3

def test_ukf_matches_kalman_on_linear():
    np.random.seed(1)
    F = np.eye(1); H = np.eye(1); Q = 0.01*np.eye(1); R = 0.5*np.eye(1)
    kf = KalmanFilter(F,H,Q,R); kf.init(np.zeros(1), np.eye(1))
    ukf = UKF(f=lambda x: F@x, h=lambda x: H@x, Q=Q, R=R, n=1)
    ukf.init(np.zeros(1), np.eye(1))
    ys = np.random.normal(5.0, np.sqrt(0.5), 50)
    for y in ys:
        kf.predict(); kf.update(np.array([y]))
        ukf.predict(); ukf.update(np.array([y]))
    assert abs(kf.x[0] - ukf.x[0]) < 0.2
