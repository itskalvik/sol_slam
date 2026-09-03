import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import message_filters
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import csv
import math

def get_yaw_from_quaternion(q):
    """
    Convert a quaternion into euler yaw angle in radians.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class SyncedDataLogger(Node):
    def __init__(self):
        super().__init__('synced_data_logger')
        
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('dvl_topic', '/a14/dvl/position_estimate')
        self.declare_parameter('output_file', 'synced_trajectory_data.csv')

        odom_topic = self.get_parameter('odom_topic').value
        dvl_topic = self.get_parameter('dvl_topic').value
        self.output_file = self.get_parameter('output_file').value

        # Initialize an empty list to hold the data in memory
        self.data_buffer = []
        
        # Setup subscribers using message_filters for synchronization
        self.odom_sub = message_filters.Subscriber(self, Odometry, odom_topic)
        self.dvl_sub = message_filters.Subscriber(
            self, PoseWithCovarianceStamped, dvl_topic, qos_profile=qos_profile_sensor_data
        )

        # Use ApproximateTimeSynchronizer in case the timestamps aren't perfectly identical
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.dvl_sub], queue_size=20, slop=0.1)
        self.ts.registerCallback(self.sync_callback)
        
        self.get_logger().info("Buffering synchronized data in memory...")
        self.get_logger().info(f"Data will be written to {self.output_file} upon node shutdown.")

    def sync_callback(self, odom_msg, dvl_msg):
        # Extract timestamp (using odom_msg as reference time)
        sec = odom_msg.header.stamp.sec
        nanosec = odom_msg.header.stamp.nanosec
        
        # Extract Sonar Odometry position[cite: 1]
        odom_x = odom_msg.pose.pose.position.x
        odom_y = odom_msg.pose.pose.position.y
        odom_z = odom_msg.pose.pose.position.z
        odom_yaw = get_yaw_from_quaternion(odom_msg.pose.pose.orientation)
        
        # Extract DVL position[cite: 2]
        dvl_x = dvl_msg.pose.pose.position.x
        dvl_y = dvl_msg.pose.pose.position.y
        dvl_z = dvl_msg.pose.pose.position.z
        dvl_yaw = get_yaw_from_quaternion(dvl_msg.pose.pose.orientation)
        
        # Append to the in-memory buffer instead of writing to disk
        self.data_buffer.append([
            sec, nanosec, 
            odom_x, odom_y, odom_z, odom_yaw,
            dvl_x, dvl_y, dvl_z, dvl_yaw
        ])
        
    def destroy_node(self):
        # Write all buffered data to disk right before the node is destroyed
        self.get_logger().info(f"Writing {len(self.data_buffer)} records to disk...")
        
        try:
            with open(self.output_file, mode='w', newline='') as f:
                csv_writer = csv.writer(f)
                
                # Write header
                csv_writer.writerow([
                    'timestamp_sec', 'timestamp_nanosec',
                    'sonar_odom_x', 'sonar_odom_y', 'sonar_odom_z', 'sonar_odom_yaw',
                    'dvl_pos_x', 'dvl_pos_y', 'dvl_pos_z', 'dvl_pos_yaw'
                ])
                
                # Write all buffered rows
                csv_writer.writerows(self.data_buffer)
                
            self.get_logger().info("Data successfully saved.")
        except Exception as e:
            self.get_logger().error(f"Failed to write data to disk: {e}")
            
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SyncedDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down synced logger node. Initiating disk write...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()