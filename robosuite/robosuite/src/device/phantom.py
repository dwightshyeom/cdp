from pynput.keyboard import Key, Listener
import numpy as np
from robosuite.devices import Device

class PhantomOmni(Device):
    """
    A custom device class for the Phantom Omni.
    """
    def __init__(self, env, ros_node, pos_sensitivity=1.0, rot_sensitivity=1.0):
        super().__init__(env)
        self.ros_node = ros_node
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity 
        
        self._reset_internal_state()

        # Keyboard listener now checks for BOTH press and release
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def _reset_internal_state(self):
        super()._reset_internal_state()
        self.last_pos = None
        self._grasp = False
        self._reset_state = 0
        self._enabled = False
        self._clutch_active = False

    def start_control(self):
        self._reset_internal_state()
        self._enabled = True

    def get_controller_state(self):
        if self.ros_node.master_position is None or not self._enabled:
            return dict(dpos=np.zeros(3), rotation=np.eye(3), raw_drotation=np.zeros(3), grasp=False, reset=self._reset_state, base_mode=0)

        current_pos = self.ros_node.master_position.copy()

        if self.last_pos is None:
            mujoco_x = 2*current_pos[2]  
            mujoco_y = 2*current_pos[0]  
            mujoco_z = 1*current_pos[1]  
            self.last_pos = np.array([mujoco_x, mujoco_y, mujoco_z])

        # Apply your calibration mapping
        mujoco_x = 2*current_pos[2]  
        mujoco_y = 2*current_pos[0]  
        mujoco_z = 1*current_pos[1]  
        mapped_pos = np.array([mujoco_x, mujoco_y, mujoco_z])
        
        # Always calculate Delta
        dpos = mapped_pos - self.last_pos

        # Update baseline for the next frame
        self.last_pos = mapped_pos

        # CLUTCH
        if not self._clutch_active:
            dpos = np.zeros(3)

        return dict(
            dpos=dpos * self.pos_sensitivity,
            rotation=np.eye(3),                       
            raw_drotation=np.zeros(3),                
            grasp=self._grasp,                    
            reset=self._reset_state,
            base_mode=0
        )

    def _postprocess_device_outputs(self, dpos, drotation):
        dpos = dpos * 125
        dpos = np.clip(dpos, -1, 1)
        return dpos, drotation

    def on_press(self, key):
        # Engage clutch when Shift is pressed down
        if key == Key.shift or key == Key.shift_l or key == Key.shift_r:
            self._clutch_active = True
            print("\nClutch Engaged\n")
        elif key == Key.esc:
            self._reset_state = 1
            self._enabled = False

    def on_release(self, key):
        # Disengage clutch when Shift is released
        if key == Key.shift or key == Key.shift_l or key == Key.shift_r:
            self._clutch_active = False
            
        # Toggle gripper on spacebar RELEASE
        elif key == Key.space:
            self._grasp = not self._grasp
            print("\nToggling gripper\n")