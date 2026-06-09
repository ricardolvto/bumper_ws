#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import math

class KalmanFilter(Node):

    def __init__(self):
        super().__init__("kalman_filter")
        self.odom_sub_ = self.create_subscription(Odometry, "bumperbot_controller/odom_noisy", self.odomCallback, 10)
        self.imu_sub_ = self.create_subscription(Imu, "imu/out", self.imuCallback, 10)
        self.odom_pub_ = self.create_publisher(Odometry, "bumperbot_controller/odom_kalman", 10)
        
        # Initially the robot has no idea about how fast is going
        self.mean_ = 0.0
        self.variance_ = 1000.0

        # Modeling the uncertainty of the sensor and the motion
        self.motion_variance_ = 4.0
        self.measurement_variance_ = 0.5

        # Store the messages - only for the orientation
        self.imu_angular_z_ = 0.0

        self.is_first_odom_ = True
        self.last_angular_z_ = 0.0
        self.motion_ = 0.0

        # Publish the filtered odometry message
        self.kalman_odom_ = Odometry()


    def odomCallback(self, odom):
        self.kalman_odom_ = odom
        self.get_logger().info(f"\n[ODOM] angular_z: {odom.twist.twist.angular.z:.4f}")

        # proteção contra NaN vindo do odom
        if math.isnan(odom.twist.twist.angular.z):
            self.get_logger().warn("ODOM is NaN — skipping callback")
            return


        if self.is_first_odom_:
            self.last_angular_z_ = odom.twist.twist.angular.z
            self.is_first_odom_ = False
            self.mean_ = odom.twist.twist.angular.z
            return
        
        # Motion
        self.motion_ = odom.twist.twist.angular.z - self.last_angular_z_
        if math.isnan(self.motion_):
            self.get_logger().warn("Motion is NaN — skipping")
            return
        
        self.get_logger().info(f"[PREDICTION] motion: {self.motion_:.4f}")
        
        # proteção extra do filtro (evita travar em NaN)
        if math.isnan(self.mean_) or math.isinf(self.mean_):
            self.get_logger().warn("Kalman state invalid — resetting")
            self.mean_ = odom.twist.twist.angular.z
            self.variance_ = 1.0

        self.get_logger().info(f"[BEFORE] mean: {self.mean_:.4f}, variance: {self.variance_:.4f}")
        self.statePrediction()
        self.get_logger().info(f"[AFTER PRED] mean: {self.mean_:.4f}, variance: {self.variance_:.4f}")
        
        self.measurementUpdate()
        self.get_logger().info(f"[AFTER UPDATE] mean: {self.mean_:.4f}, variance: {self.variance_:.4f}")

        # Update for the next iteration
        self.last_angular_z_ = odom.twist.twist.angular.z

        # Update and publish the filtered odom message
        self.kalman_odom_.twist.twist.angular.z = self.mean_
        self.get_logger().info(f"[OUTPUT] filtered angular_z: {self.mean_:.4f}")
        self.odom_pub_.publish(self.kalman_odom_)


    def imuCallback(self, imu):
        # Store the measurement update
        self.imu_angular_z_ = imu.angular_velocity.z
        self.get_logger().info(f"[IMU] angular_z: {self.imu_angular_z_:.4f}")


    def measurementUpdate(self):
        self.mean_ = (self.measurement_variance_ * self.mean_ + self.variance_ * self.imu_angular_z_) \
                   / (self.variance_ + self.measurement_variance_)
                     
        self.variance_ = (self.variance_ * self.measurement_variance_) \
                       / (self.variance_ + self.measurement_variance_)


    def statePrediction(self):
        self.mean_ = self.mean_ + self.motion_
        self.variance_ = self.variance_ + self.motion_variance_


def main():
    rclpy.init()

    kalman_filter = KalmanFilter()
    rclpy.spin(kalman_filter)
    
    kalman_filter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()