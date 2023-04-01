import pandas as pd
import numpy as np

from scipy import stats


chat_id = 973327975 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    n = len(x)
    alpha = 1 - p
    alpha_star = alpha / 2
    chi2_left = stats.chi2.ppf(alpha_star, 2 * n)
    chi2_right = stats.chi2.ppf(1 - alpha_star, 2 * n)
    b_left = np.max(x) * (1 + (n / chi2_right))**(-1/ n)
    b_right = np.max(x) * (1 + (n / chi2_left))**(-1/ n)
    return (b_left, b_right)
