#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathThroughPoses


class TspSolverActionServer(Node):

    def __init__(self):
        super().__init__('tsp_solver_node')

        self.get_logger().info('Initializing TSP Solver Action Server for ROS 2 Jazzy...')

        self._callback_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self,
            ComputePathThroughPoses,
            'compute_tsp_route',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._callback_group
        )

        self.get_logger().info('TSP Solver Action Server is ready on topic: /compute_tsp_route')

    def goal_callback(self, goal_request):
        if len(goal_request.goals) < 2:
            self.get_logger().warn('Rejected goal: At least 2 poses are required to compute a TSP route.')
            return GoalResponse.REJECT

        self.get_logger().info(f'Received a TSP path optimization request with {len(goal_request.goals)} waypoints.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received request to cancel TSP computation.')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing TSP optimization...')

        feedback_msg = ComputePathThroughPoses.Feedback()
        result = ComputePathThroughPoses.Result()

        input_poses = goal_handle.request.goals
        coords = np.array([[p.pose.position.x, p.pose.position.y] for p in input_poses])

        # Nearest Neighbor initial tour
        num_nodes = len(coords)
        unvisited = list(range(1, num_nodes))
        current_node = 0
        tour = [current_node]

        while unvisited:
            last = tour[-1]
            next_node = min(unvisited, key=lambda x: np.linalg.norm(coords[last] - coords[x]))
            unvisited.remove(next_node)
            tour.append(next_node)

        # 2-Opt refinement
        tour = self._two_opt(coords, tour)

        optimized_poses = [input_poses[idx] for idx in tour]

        goal_handle.succeed()

        result.path.poses = optimized_poses
        result.path.header.stamp = self.get_clock().now().to_msg()
        result.path.header.frame_id = "map"

        self.get_logger().info('TSP optimization complete. Optimized path generated successfully.')
        return result

    def _two_opt(self, coords, tour):
        best_tour = tour
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_tour) - 2):
                for j in range(i + 1, len(best_tour)):
                    if j - i == 1:
                        continue
                    new_tour = best_tour[:]
                    new_tour[i:j] = best_tour[j-1:i-1:-1]
                    if self._calculate_total_distance(coords, new_tour) < self._calculate_total_distance(coords, best_tour):
                        best_tour = new_tour
                        improved = True
            if not rclpy.ok():
                break
        return best_tour

    def _calculate_total_distance(self, coords, tour):
        distance = 0.0
        for i in range(len(tour) - 1):
            distance += np.linalg.norm(coords[tour[i]] - coords[tour[i+1]])
        return distance


def main(args=None):
    rclpy.init(args=args)
    node = TspSolverActionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
