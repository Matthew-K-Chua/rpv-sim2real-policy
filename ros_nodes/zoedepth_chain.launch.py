"""ROS side of the monocular-depth pipeline: receiver [-> cloud [-> scan]].

    ros2 launch ros_nodes/zoedepth_chain.launch.py \
        zoedepth_url:=http://127.0.0.1:12185/zoedepth scale_align:=true

Start the MODEL first (scripts/launch_zoedepth_server.sh, conda env, GPU). This
file starts only ROS-side things, in ROS's own Python 3.10 environment. Design
and the reasoning behind every default here: vlfm/vlm/zoedepth.py.

THE THREE STAGES ARE SEPARATELY SWITCHED, AND THE DEFAULTS ARE NOT "ALL ON":

  1. zoedepth_depth_node          ALWAYS. Publishes /zoedepth/image_raw (16UC1)
                                  + /zoedepth/camera_info. Costs one GPU forward
                                  pass per frame and nothing else in ROS reads it
                                  unless you point something at it.
  2. publish_cloud   (default ON) depth_image_proc -> /zoedepth/points. Inert
                                  until a costmap names it as an observation
                                  source; it exists so you can LOOK at the cloud
                                  in RViz before trusting it with anything.
  3. publish_scan    (default OFF) pointcloud_to_laserscan -> /scan_parts/d435,
                                  which can be folded into /scan --
                                  the topic SLAM AND BOTH COSTMAPS read.

Stage 3 is off because of an asymmetry, not out of caution in general. The merge
is a bin-wise MINIMUM: a monocular OVER-estimate is harmlessly discarded wherever
one of Spot's real cameras sees closer, but a monocular UNDER-estimate WINS the
minimum and becomes a phantom obstacle in the map slam_toolbox is building. On
Spot, where /scan is already 360 degrees of real depth, that is a bad trade for a
forward 87 degree wedge that is already the best-covered part of the circle.

Turn it on when the camera is the ONLY depth source (the TurtleBot 4 rig), or
deliberately and temporarily, watching /zoedepth/diagnostics.

NAV2: this file does not touch the costmaps. Adding /zoedepth/points as an
observation source is an edit to config/nav2_spot.yaml in the CEAI repo -- see
vlfm/vlm/zoedepth.py for the settings that make it safe (short
obstacle_max_range, LOCAL costmap only).
"""
import math
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

# Absolute path to the receiver, resolved from this file so the launch works when
# run by path. Nothing in this repo is installed as a ROS package.
NODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zoedepth_depth_node.py")

# MUST match rpv-ros2-bridge's launch/robot_onboard.launch.py exactly. A scan merger REFUSES an
# input whose grid differs -- that is deliberate, and it is what makes a sixth
# contributor a no-op change to the merger. If those constants ever move, these
# move with them.
SCAN_ANGLE_MIN = -math.pi
SCAN_ANGLE_MAX = math.pi
SCAN_ANGLE_INCREMENT = math.pi / 360.0


def generate_launch_description() -> LaunchDescription:
    args = [
        ("zoedepth_url", "http://127.0.0.1:12185/zoedepth", "the model server started by scripts/launch_zoedepth_server.sh"),
        ("rgb_topic", "/camera/camera/color/image_raw/compressed",
         "the Pi's colour stream. Older realsense2_camera uses the flat /camera/color/... layout"),
        ("camera_info_topic", "/camera/camera/color/camera_info", "must be the COLOUR info: the depth is registered to it"),
        ("out_namespace", "/zoedepth", "where the depth image, camera_info and diagnostics are published"),
        ("trust_max_m", "8.0",
         "beyond this the model's guess is published as INVALID (0). The NYU head's own "
         "architectural ceiling is 10 m (bin_configurations in the checkpoint's config.json)"),
        ("trust_min_m", "0.3", "below this, invalid -- the robot's own body and the lens minimum"),
        ("max_rate_hz", "6.0", "matches the Pi's colour cap; stops a faster stream saturating the GPU"),
        ("scale_align", "false", "fit ZoeDepth's scale against /scan every frame (fits scale against /scan)"),
        ("scan_topic", "/scan", "the metric reference for scale alignment"),
        ("lidar_frame", "body", "pointcloud_to_laserscan's target_frame in robot_onboard.launch.py"),
        ("camera_optical_frame", "camera_color_optical_frame", "needs the six MEASURED d435_* launch arguments"),
        ("publish_cloud", "true", "stage 2 -- /zoedepth/points, for RViz and (opt-in) a Nav2 observation source"),
        ("publish_scan", "false", "stage 3 -- /scan_parts/d435 into the merger. READ THIS FILE'S HEADER FIRST"),
    ]
    ld = LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args])

    # package=None: `executable` is then taken as a path, which is how a node
    # that lives in a non-ROS repo gets launched without being colcon-installed.
    ld.add_action(
        Node(
            package=None,
            executable=NODE,
            name="zoedepth_depth",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "zoedepth_url": LaunchConfiguration("zoedepth_url"),
                "rgb_topic": LaunchConfiguration("rgb_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "out_namespace": LaunchConfiguration("out_namespace"),
                "trust_max_m": LaunchConfiguration("trust_max_m"),
                "trust_min_m": LaunchConfiguration("trust_min_m"),
                "max_rate_hz": LaunchConfiguration("max_rate_hz"),
                "scale_align": LaunchConfiguration("scale_align"),
                "scan_topic": LaunchConfiguration("scan_topic"),
                "lidar_frame": LaunchConfiguration("lidar_frame"),
                "camera_optical_frame": LaunchConfiguration("camera_optical_frame"),
            }],
        )
    )

    # --- stage 2: depth raster -> point cloud --------------------------------
    # PointCloudXyzNode, not Xyzrgb: the colour frame is right there, but a
    # colour cloud is ~4x the bytes and neither the costmaps nor RViz's obstacle
    # view need it. Same choice robot_onboard.launch.py makes for the D435.
    ld.add_action(
        ComposableNodeContainer(
            name="zoedepth_cloud_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            condition=IfCondition(LaunchConfiguration("publish_cloud")),
            composable_node_descriptions=[
                ComposableNode(
                    package="depth_image_proc",
                    plugin="depth_image_proc::PointCloudXyzNode",
                    name="zoedepth_points",
                    remappings=[
                        ("image_rect", [LaunchConfiguration("out_namespace"), "/image_raw"]),
                        ("camera_info", [LaunchConfiguration("out_namespace"), "/camera_info"]),
                        ("points", [LaunchConfiguration("out_namespace"), "/points"]),
                    ],
                )
            ],
            output="screen",
        )
    )

    # --- stage 3: cloud -> a wedge of /scan ----------------------------------
    # OFF by default. See the header. The height band and range band below are
    # deliberately NOT copies of robot_onboard.launch.py's:
    #   * min_height 0.05 (not -0.30): that file's band is set to exclude SPOT'S
    #     OWN LEGS from its body cameras. This camera cannot see Spot's legs, and
    #     a band starting below the floor plane would turn the model's floor --
    #     the thing it estimates most confidently and most densely -- into a wall
    #     of obstacles directly in front of the robot.
    #   * range_max 3.0 (not 4.0): the mono far field is where scale error is
    #     largest, and a bin-wise minimum gives an under-estimate the last word.
    ld.add_action(
        Node(
            package="pointcloud_to_laserscan",
            executable="pointcloud_to_laserscan_node",
            name="scan_from_zoedepth",
            condition=IfCondition(LaunchConfiguration("publish_scan")),
            remappings=[
                ("cloud_in", [LaunchConfiguration("out_namespace"), "/points"]),
                ("scan", "/scan_parts/d435"),
            ],
            parameters=[{
                "target_frame": LaunchConfiguration("lidar_frame"),
                "transform_tolerance": 0.05,
                "min_height": 0.05,
                "max_height": 0.50,
                # Shared grid -- the merger refuses anything else.
                "angle_min": SCAN_ANGLE_MIN,
                "angle_max": SCAN_ANGLE_MAX,
                "angle_increment": SCAN_ANGLE_INCREMENT,
                "scan_time": 1.0 / 6.0,          # the Pi's colour rate, not 15 Hz
                "range_min": 0.3,
                "range_max": 3.0,
                "use_inf": True,
                "inf_epsilon": 1.0,
            }],
            output="screen",
        )
    )
    return ld
