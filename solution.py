import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 973327975 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    alpha = 1 - p
    b_hat = x.max() + 0.02
    n = len(x)
    s = np.sqrt(((x - b_hat + 0.02) ** 2).sum() / (n - 1))
    z = norm.ppf(1 - alpha / 2)
    left = b_hat - z * s / np.sqrt(n)
    right = b_hat + z * s / np.sqrt(n)
    return left, right
