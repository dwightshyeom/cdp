import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import transforms3d as tf3d

class TeleopNode(Node):
    def __init__(self):
        super().__init__("teleop_node")
        self.get_logger().info("TeleopNode has been initialized.")
        self.create_subscription(PoseStamped, "/arm/measured_cp", self.master_pose_callback, 10)
        
        self.master_position = None
        self.master_quat = None
        self.master_R = None

    def master_pose_callback(self, msg: PoseStamped):
        self.master_position = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        ])
        self.master_quat = np.array([
            msg.pose.orientation.w, msg.pose.orientation.x, 
            msg.pose.orientation.y, msg.pose.orientation.z
        ])
        self.master_R = tf3d.quaternions.quat2mat(self.master_quat)