import h5py
import numpy as np

# RAW dataset (Where the attribute is safely stored)
raw_dataset_path = "../robosuite/custom_dataset/demo.hdf5"

# FINAL dataset (Where we need to inject the goal array)
final_dataset_path = "../robosuite/custom_dataset/low_dim_boxpush_120.hdf5"

f_raw = h5py.File(raw_dataset_path, "r")
f_final = h5py.File(final_dataset_path, "a")

for ep in f_final["data"]:
    ep_grp_final = f_final["data"][ep]
    ep_grp_raw = f_raw["data"][ep]
    
    # 1. Read the smuggled attribute from the RAW file
    if "target_push_distance" in ep_grp_raw.attrs:
        target_dist = ep_grp_raw.attrs["target_push_distance"]
    else:
        print(f"Warning: No condition found for {ep} in raw file.")
        continue
        
    # 2. Match the exact timesteps of this episode
    timesteps = ep_grp_final["actions"].shape[0]
    
    # 3. Create the flat condition array
    cond_array = np.full((timesteps, 1), target_dist, dtype=np.float32)
    
    # 4. Inject the condition array into the FINAL dataset under obs/push_distance
    if "push_distance" in ep_grp_final["obs"]:
        del ep_grp_final["obs"]["push_distance"]
        
    ep_grp_final["obs"].create_dataset("push_distance", data=cond_array)
    
    print(f"[{ep}] Injected obs/push_distance -> Value: {target_dist} | Shape: {cond_array.shape}")
    
f_raw.close()
f_final.close()
print("\n[✓] Goal conditions successfully injected!")