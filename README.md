# ROS_Package_example
## I. Installation
1. Ubuntu 16.04 or newer
2. One of these following version of [ROS](https://ros.org)
    
      $ sudo apt-get install ros-kinetic-desktop-full
      ```
3. Create catkin workspace
    ```
    $ mkdir -p ~/catkin_ws/src
    $ cd ~/catkin_ws/
    $ catkin_make
    $ echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
    $ source ~/.bashrc
    ```
  
4. Install rosbridge-suite
    ```
    $ sudo apt-get install ros-kinetic-rosbridge-server
    ```
5. [New version unity](https://drive.google.com/drive/u/0/folders/1ShsdXU_2Dk86wIaTQb9mbJNWgT3kEy0_?hl=en)
# ROS-Self-Car

6. catkin_make
7. source devel/setup.bash
8. Chay code fpt: roslaunch lane_detect lane_detect.launch
    Sau do chay unity 
9. Chay code psa: roslaunch psa psa.launch
    Sau do chay unity 
