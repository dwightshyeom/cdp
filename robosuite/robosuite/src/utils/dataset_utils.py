import os
import h5py
import datetime
import numpy as np
from glob import glob
import robosuite as suite

def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """Compiles temporary robosuite npz files into a robomimic-compatible hdf5 file."""
    os.makedirs(out_dir, exist_ok=True)
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")

    num_eps = 0
    env_name = None 

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states, actions = [], []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if not states: continue

        if success:
            del states[-1] 
            assert len(states) == len(actions)
            num_eps += 1
            ep_data_grp = grp.create_group(f"demo_{num_eps}")
            
            with open(os.path.join(directory, ep_directory, "model.xml"), "r") as xml_f:
                ep_data_grp.attrs["model_file"] = xml_f.read()

            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))

            # store the target distance condition
            cond_path = os.path.join(directory, ep_directory, "condition.npy")
            if os.path.exists(cond_path):
                target_dist = np.load(cond_path)[0]
                print(f"[{ep_directory}] Found condition.npy with target distance: {target_dist}")
                ep_data_grp.attrs["target_push_distance"] = float(target_dist)

    now = datetime.datetime.now()
    grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
    grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name if env_name else "Lift"
    grp.attrs["env_info"] = env_info
    
    f.close()
    print(f"\nFinal dataset saved to: {hdf5_path}")