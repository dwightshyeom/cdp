import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask

from robosuite.utils.observables import Observable, sensor
import robosuite.utils.transform_utils as T

class BoxPush(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1e-5, 5e-5, 1e-5),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=512,
        camera_widths=512,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        
        # The default condition (Will be overwritten by pipeline)
        self.target_push_distance = 0.1 
        self.start_x = 0.0 
        self.start_y = 0.0

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    # Pipeline Setter Method
    def set_target_distance(self, distance):
        """Called by the data collection loop before env.reset()"""
        self.target_push_distance = distance

    def _load_model(self):
        super()._load_model()

        # Robot base position
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](2.0)
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        
        self.arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=(0, 0, 0.8),
        )

        self.box_red = BoxObject(
            name="cube_red", 
            size=[0.03, 0.03, 0.03], 
            rgba=[1, 0, 0, 1],
            friction=[1e-1, 1, 1],
        )
        
        # The 3 visual indicator bars
        self.bar_05 = BoxObject(name="bar_05", size=[0.35, 0.005, 0.001], rgba=[0, 1, 0, 0.4], joints=None, obj_type="visual")
        self.bar_10 = BoxObject(name="bar_10", size=[0.35, 0.005, 0.001], rgba=[0, 0, 1, 0.4], joints=None, obj_type="visual")
        self.bar_15 = BoxObject(name="bar_15", size=[0.35, 0.005, 0.001], rgba=[1, 1, 0, 0.4], joints=None, obj_type="visual")

        for bar in [self.bar_05, self.bar_10, self.bar_15]:
            self.arena.worldbody.append(bar.get_obj())

        self.model = ManipulationTask(
            mujoco_arena=self.arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.box_red] 
        )

    def _reset_internal(self):
        super()._reset_internal()
        table_z = self.arena.table_top_abs[2]

        # Place the visual indicator bars
        distances = [0.1, 0.2, 0.3]
        bars = [self.bar_05, self.bar_10, self.bar_15]
        
        for bar, dist in zip(bars, distances):
            pos_x = self.start_x + dist
            bar_pos = np.array([0.0, pos_x, table_z + 0.001])
            body_id = self.sim.model.body_name2id(bar.root_body)
            self.sim.model.body_pos[body_id] = bar_pos

        box_half_height = 0.03
        box_pos = np.array([self.start_x, self.start_y, table_z + box_half_height])
        default_quat = np.array([1, 0, 0, 0])
        
        self.sim.data.set_joint_qpos(self.box_red.joints[0], np.concatenate([box_pos, default_quat]))
        self.sim.forward()

    def _setup_references(self):
        super()._setup_references()
        self.box_body_id = self.sim.model.body_name2id(self.box_red.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality="object")
        def box_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.box_body_id])

        @sensor(modality="object")
        def gripper_to_box_pos(obs_cache):
            return obs_cache["box_pos"] - obs_cache[f"{pf}eef_pos"] if \
                "box_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

        @sensor(modality="goal")
        def push_distance(obs_cache):
            return np.array([self.target_push_distance], dtype=np.float32)

        sensors = [box_pos, gripper_to_box_pos, push_distance]
        names = ["box_pos", "gripper_to_box_pos", "push_distance"]

        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def reward(self, action=None):
        return 0 
            
    def _check_success(self):
        target_y = self.start_y + self.target_push_distance
        box_pos = self.sim.data.body_xpos[self.box_body_id]
        y_dist_to_target = abs(box_pos[1] - target_y)
        x_dist_from_center = abs(box_pos[0] - self.start_x)
        
        # Tolerance
        reached_distance = y_dist_to_target < 0.01 # Critical tolerance
        stayed_on_path = x_dist_from_center < 0.05
        stayed_on_path_z = abs(box_pos[2] - self.arena.table_top_abs[2] - 0.03) < 0.01 # Not lifted

        return reached_distance and stayed_on_path and stayed_on_path_z