import numpy as np
def logScore(y_obs, y_pred):
    likelihoodElements= (y_obs*np.log(y_pred) + (1 - y_obs)*np.log(1 - y_pred))
    logScore = np.mean(likelihoodElements)
    return logScore.values[0]