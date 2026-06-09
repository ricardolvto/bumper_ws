#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateToPose


class WaypointFollower(Node):

    def __init__(self):
        super().__init__("waypoint_follower")
        self.action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.subscription = self.create_subscription(
            Path, "unordered_waypoints", self.waypoints_callback, 10
        )
        self.waypoints = []
        self.current_index = 0

    def waypoints_callback(self, msg):
        if not msg.poses:
            self.get_logger().warn("Received empty waypoint list, ignoring.")
            return
        self.waypoints = msg.poses
        self.current_index = 0
        self.get_logger().info(f"Received {len(self.waypoints)} waypoints. Starting navigation.")
        self.send_next_goal()

    def send_next_goal(self):
        if self.current_index >= len(self.waypoints):
            self.get_logger().info("All waypoints reached!")
            return

        pose = self.waypoints[self.current_index]
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose = pose.pose

        self.get_logger().info(
            f"Navigating to waypoint {self.current_index + 1}/{len(self.waypoints)}: "
            f"x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}"
        )

        self.action_client.wait_for_server()
        send_goal_future = self.action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(
                f"Waypoint {self.current_index + 1} rejected, skipping."
            )
            self.current_index += 1
            self.send_next_goal()
            return

        self.get_logger().info(f"Waypoint {self.current_index + 1} accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result()
        status = result.status
        # 4 = SUCCEEDED, 6 = ABORTED, 5 = CANCELED
        if status == 4:
            self.get_logger().info(
                f"Waypoint {self.current_index + 1} reached successfully."
            )
        else:
            self.get_logger().warn(
                f"Waypoint {self.current_index + 1} failed with status {status}, moving on."
            )
        self.current_index += 1
        self.send_next_goal()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
