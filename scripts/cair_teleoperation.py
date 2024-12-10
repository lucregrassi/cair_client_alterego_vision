import rospy
from geometry_msgs.msg import Twist
import socket
import threading


class CairTeleoperation:
    def __init__(self, command_port=54321):
        # ROS publisher
        self.pub = rospy.Publisher('/robot_alterego6/cmd_vel', Twist, queue_size=10)
        rospy.init_node('cair_teleoperation', anonymous=True)

        # UDP configuration
        self.command_port = command_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('', self.command_port))
        self.running = False

    def start_udp_listener(self):
        self.running = True
        thread = threading.Thread(target=self.udp_listener)
        thread.start()

    def udp_listener(self):
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                command = data.decode('utf-8').strip()
                rospy.loginfo(f"Received command: {command}")
                self.handle_command(command)
            except Exception as e:
                rospy.logerr(f"Error in UDP listener: {e}")

    def handle_command(self, command):
        twist_msg = Twist()
        if command == "MOVE_FORWARD":
            twist_msg.linear.x = 1
        elif command == "MOVE_BACKWARD":
            twist_msg.linear.x = -1
        elif command == "ROTATE_LEFT":
            twist_msg.angular.z = 0.3
        elif command == "ROTATE_RIGHT":
            twist_msg.angular.z = -0.3
        elif command == "STOP":
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
        else:
            rospy.logwarn(f"Unknown command: {command}")
            return

        # Publish the Twist message
        self.pub.publish(twist_msg)
        rospy.loginfo(f"Published message: {twist_msg}")

    def stop_udp_listener(self):
        self.running = False
        self.socket.close()


if __name__ == "__main__":
    manager = CairTeleoperation()
    try:
        rospy.loginfo("Starting UDP listener...")
        manager.start_udp_listener()
        rospy.spin()  # Keep the ROS node running
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down CairTeleoperation.")
        manager.stop_udp_listener()
