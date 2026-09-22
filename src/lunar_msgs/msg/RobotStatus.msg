# 单台机器人状态
string robot_id
string role              # SCOUT | EXCAVATOR | HAULER
geometry_msgs/Pose pose
float32 battery          # 剩余能量 0..1
float32 cargo            # 当前载量（units）
string state             # IDLE|MOVING|SURVEYING|EXCAVATING|LOADING|HAULING|DUMPING
