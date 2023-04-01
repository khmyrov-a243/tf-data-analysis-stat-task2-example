import pandas as pd
import numpy as np

from scipy.stats import uniform


chat_id = 973327975 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    # Находим квантиль распределения для заданного уровня доверия
    q = 1 - (1 - p) / 2
    u = uniform.ppf(q, loc=0.02)

    # Вычисляем точечную оценку для b
    b_hat = np.max(x)

    # Вычисляем стандартную ошибку
    se = (b_hat - 0.02) / uniform.ppf(q, loc=0.02, scale=b_hat - 0.02)

    # Вычисляем границы доверительного интервала
    left = b_hat - se * uniform.ppf(1 - (1 - p) / 2, loc=0, scale=1)
    right = b_hat - se * uniform.ppf((1 - p) / 2, loc=0, scale=1)

    return left, right
