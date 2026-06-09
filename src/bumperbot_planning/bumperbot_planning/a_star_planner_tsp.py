#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import Buffer, TransformListener, LookupException
from queue import PriorityQueue

from nav2_msgs.action import ComputePathThroughPoses, NavigateThroughPoses


class GraphNode:
    def __init__(self, x, y, cost=0, heuristic=0, prev=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.heuristic = heuristic
        self.prev = prev

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        return GraphNode(self.x + other[0], self.y + other[1])


class AStarPlanner(Node):
    def __init__(self):
        super().__init__("a_star_node")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(depth=10)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, map_qos
        )

        self.goals_sub = self.create_subscription(
            Path, "/unordered_waypoints", self.waypoints_callback, 10
        )

        self.path_pub = self.create_publisher(Path, "/a_star/path", 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "/a_star/visited_map", 10)

        self._tsp_client = ActionClient(self, ComputePathThroughPoses, 'compute_tsp_route')

        # Action client to actually drive the robot through the sorted waypoints
        self._nav_client = ActionClient(self, NavigateThroughPoses, 'navigate_through_poses')

        self.map_ = None
        self.visited_map_ = OccupancyGrid()

    def map_callback(self, map_msg: OccupancyGrid):
        self.map_ = map_msg
        self.visited_map_.header.frame_id = map_msg.header.frame_id
        self.visited_map_.info = map_msg.info
        self.visited_map_.data = [-1] * (map_msg.info.height * map_msg.info.width)

    def waypoints_callback(self, msg: Path):
        if self.map_ is None:
            self.get_logger().error("No map received yet!")
            return

        if len(msg.poses) < 2:
            self.get_logger().warn("Need at least 2 waypoints to process a route optimization.")
            return

        self.get_logger().info(f"Received {len(msg.poses)} waypoints. Sending to TSP solver...")

        if not self._tsp_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("TSP solver action server not available!")
            return

        goal_msg = ComputePathThroughPoses.Goal()
        goal_msg.goals = msg.poses

        send_goal_future = self._tsp_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.tsp_response_callback)

    def tsp_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("TSP Solver rejected the waypoints request.")
            return

        self.get_logger().info("TSP Solver accepted request. Processing optimal route calculation...")
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.tsp_result_callback)

    def tsp_result_callback(self, future):
        result = future.result().result
        sorted_poses = result.path.poses
        self.get_logger().info(f"Received optimized order of {len(sorted_poses)} nodes from TSP.")

        try:
            map_to_base_tf = self.tf_buffer.lookup_transform(
                self.map_.header.frame_id, "base_footprint", rclpy.time.Time()
            )
            start_pose = Pose()
            start_pose.position.x = map_to_base_tf.transform.translation.x
            start_pose.position.y = map_to_base_tf.transform.translation.y
            start_pose.orientation = map_to_base_tf.transform.rotation
        except LookupException:
            self.get_logger().warn("Could not find robot base_footprint. Defaulting path from first waypoint.")
            start_pose = sorted_poses[0].pose

        # Build A* path for visualization on /a_star/path
        final_complete_path = Path()
        final_complete_path.header.frame_id = self.map_.header.frame_id

        current_start = start_pose
        for target_pose_stamped in sorted_poses:
            self.visited_map_.data = [-1] * (self.visited_map_.info.height * self.visited_map_.info.width)
            segment_path = self.plan(current_start, target_pose_stamped.pose)
            final_complete_path.poses.extend(segment_path.poses)
            current_start = target_pose_stamped.pose

        if final_complete_path.poses:
            self.get_logger().info("Complete optimized A* mission path generated — publishing for visualization.")
            self.path_pub.publish(final_complete_path)
        else:
            self.get_logger().error("Failed to generate complete path segments.")
            return

        # Send TSP-sorted poses to Nav2 to actually drive the robot
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("navigate_through_poses action server not available!")
            return

        # Ensure every pose has a valid frame_id and timestamp — Nav2 requires this for transforms
        now = self.get_clock().now().to_msg()
        for pose in sorted_poses:
            pose.header.frame_id = "map"
            pose.header.stamp = now

        nav_goal = NavigateThroughPoses.Goal()
        nav_goal.poses = sorted_poses
        self.get_logger().info(f"Sending {len(sorted_poses)} sorted waypoints to Nav2 for execution...")
        nav_future = self._nav_client.send_goal_async(nav_goal)
        nav_future.add_done_callback(self.nav_response_callback)

    def nav_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Nav2 rejected the navigation goal.")
            return
        self.get_logger().info("Nav2 accepted navigation goal. Robot is moving!")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)

    def nav_result_callback(self, future):
        status = future.result().status
        if status == 4:
            self.get_logger().info("Robot successfully completed all waypoints!")
        else:
            self.get_logger().warn(f"Navigation ended with status {status}.")

    def plan(self, start: Pose, goal: Pose):
        explore_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        pending_nodes = PriorityQueue()
        visited_nodes = set()

        start_node = self.world_to_grid(start)
        goal_node = self.world_to_grid(goal)
        start_node.heuristic = self.manhattan_distance(start_node, goal_node)
        pending_nodes.put(start_node)

        active_node = start_node
        while not pending_nodes.empty() and rclpy.ok():
            active_node = pending_nodes.get()

            if active_node == goal_node:
                break

            for dir_x, dir_y in explore_directions:
                new_node: GraphNode = active_node + (dir_x, dir_y)

                if (new_node not in visited_nodes and self.pose_on_map(new_node) and
                    self.map_.data[self.pose_to_cell(new_node)] == 0):

                    new_node.cost = active_node.cost + 1
                    new_node.heuristic = self.manhattan_distance(new_node, goal_node)
                    new_node.prev = active_node

                    pending_nodes.put(new_node)
                    visited_nodes.add(new_node)

            self.visited_map_.data[self.pose_to_cell(active_node)] = -106

        path = Path()
        path.header.frame_id = self.map_.header.frame_id
        while active_node and active_node.prev and rclpy.ok():
            last_pose: Pose = self.grid_to_world(active_node)
            last_pose_stamped = PoseStamped()
            last_pose_stamped.header.frame_id = self.map_.header.frame_id
            last_pose_stamped.pose = last_pose
            path.poses.append(last_pose_stamped)
            active_node = active_node.prev

        path.poses.reverse()
        return path

    def manhattan_distance(self, node: GraphNode, goal_node: GraphNode):
        return abs(node.x - goal_node.x) + abs(node.y - goal_node.y)

    def pose_on_map(self, node: GraphNode):
        return 0 <= node.x < self.map_.info.width and 0 <= node.y < self.map_.info.height

    def world_to_grid(self, pose: Pose) -> GraphNode:
        grid_x = int((pose.position.x - self.map_.info.origin.position.x) / self.map_.info.resolution)
        grid_y = int((pose.position.y - self.map_.info.origin.position.y) / self.map_.info.resolution)
        return GraphNode(grid_x, grid_y)

    def grid_to_world(self, node: GraphNode) -> Pose:
        pose = Pose()
        pose.position.x = node.x * self.map_.info.resolution + self.map_.info.origin.position.x
        pose.position.y = node.y * self.map_.info.resolution + self.map_.info.origin.position.y
        return pose

    def pose_to_cell(self, node: GraphNode):
        return node.y * self.map_.info.width + node.x


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
