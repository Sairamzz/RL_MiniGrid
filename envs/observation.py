import numpy as np


def encode_obs_tabular(obs): # Returns a tuple representation of the observation
    if isinstance(obs, dict) and "image" in obs:
        image = np.array(obs["image"]) 
        direction = int(obs.get("direction", 0))
        
        if image.ndim == 3:
            obj = image[:, :, 0] # Object type
        else:
            obj = image

        h, w = obj.shape
        ch, cw = h // 2, w // 2

        center = obj[max(0, ch-1): ch+2, max(0, cw-1): cw+2]

        return (tuple(center.flatten().tolist()), direction) # We return a tuple of the center 3x3 grid and the agent's direction
    
    arr = np.array(obs) # If the observation is not a dictionary with an image key, we just flatten it and convert to a tuple
    return tuple(arr.flatten().tolist())

def obs_to_history_key(obs): # Converts the observation to a key for the history dictionary used in POMCP
    if isinstance(obs, dict) and "image" in obs:
        image = np.array(obs["image"])
        direction = int(obs.get("direction", 0))

        return (tuple(image.flatten().tolist()), direction) # We return a tuple of the entire image and the agent's direction
    
    arr = np.array(obs) # If the observation is not a dictionary with an image key, we just flatten it and convert to a tuple
    return tuple(arr.flatten().tolist())