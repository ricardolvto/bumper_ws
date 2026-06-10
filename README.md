

*Terminal 1:*
colcon build
source install/setup.bash
ros2 launch bumperbot_bringup simulated_robot.launch.py

*Terminal 2: TSP solver*
source install/setup.bash
ros2 run bumperbot_planning tsp_solver_node.py

*Terminal 3: Astar + TSP planner:*
source install/setup.bash
ros2 run bumperbot_planning a_star_planner_tsp.py

*Terminal 4 — Send random waypoints:*
ros2 topic pub --once /unordered_waypoints nav_msgs/msg/Path "$(python3 -c '
import json, random

poses = []
for _ in range(10):
    # Use smaller range near map center where free space exists
    x = random.uniform(-3.0, 3.0)
    y = random.uniform(-3.0, 3.0)
    poses.append({
        "header": {"frame_id": "map"},
        "pose": {
            "position": {"x": round(x, 2), "y": round(y, 2), "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        }
    })

print(json.dumps({"header": {"frame_id": "map"}, "poses": poses}))
')"
