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
        image = np.array(obs["image"], dtype=np.int32)
        direction = int(obs.get("direction", 0))
        return (tuple(image.flatten().tolist()), direction) # We return a tuple of the object type and the agent's direction
    
    arr = np.array(obs) # If the observation is not a dictionary with an image key, we just flatten it and convert to a tuple
    return tuple(arr.flatten().tolist())

def encode_obs_ppo(obs): # Flatten MiniGrid observation into a vector for PPO
    image = np.array(obs["image"], dtype=np.int64)

    obj = image[:, :, 0]
    color = image[:, :, 1]
    state = image[:, :, 2]
    direction = int(obs.get("direction", 0))

    obj_oh = np.eye(11, dtype=np.float32)[np.clip(obj, 0, 10)]
    color_oh = np.eye(6, dtype=np.float32)[np.clip(color, 0, 5)]
    state_oh = np.eye(3, dtype=np.float32)[np.clip(state, 0, 2)]

    image_oh = np.concatenate([obj_oh, color_oh, state_oh], axis=-1)
    image_flat = image_oh.reshape(-1)

    dir_oh = np.eye(4, dtype=np.float32)[np.clip(direction, 0, 3)]

    return np.concatenate([image_flat, dir_oh], axis=0)