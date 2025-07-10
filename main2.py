
import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

_pairs_params = [(48,49,3.22496983496616,0.333467256193952)]


def compute_zscore(spread_arr: np.ndarray):
    mean = np.mean(spread_arr)
    std = np.std(spread_arr)

    z_score = (spread_arr - mean) / std

    return z_score

def current_z(spread: np.ndarray) -> float:
    """
    Return the z-score of the most recent spread value.
    """
    zscores = compute_zscore(spread)
    return float(zscores[-1])

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Called each day with prcSoFar (nInst × nt).
    Caches top-5 pairs+hedge on first call, then every day:
      1) computes each spread = log Pi – (β·log Pj + α),
      2) z-scores it, 
      3) if z>2 → short spread, if z<–2 → long spread, else flat.
    """
    global currentPos, _pairs_params

    nins, nt = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins, dtype=int)


    # 2) Each day: update positions for each cached pair
    for i, j, α, β in _pairs_params:
        pi = prcSoFar[i]       # shape (nt,)
        pj = prcSoFar[j]
        spread = np.log(pi) - (β * np.log(pj) + α)
        z      = current_z(spread)

        if   z >  2.0:
            currentPos[i], currentPos[j] = -10, +10
        elif z < -2.0:
            currentPos[i], currentPos[j] = +10, -10
        else:
            currentPos[i], currentPos[j] =    0,   0

    return currentPos
