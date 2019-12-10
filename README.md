# bwi_segbot_hallway

(1) Setting up the BWI project
Follow the instructions in the given link
https://github.com/utexas-bwi/documentation/wiki/Making-a-BWI-Workspace

(2) Clone bwi folder from the following link
https://github.com/ub31/bwi
Replace the bwi folder in the BWI workspace.

(3) Clone the current bwi_segbot_hallway repository under bwi_common folder and do a catkin build.

After a successful build, for launching the training script :
  cd ~/catkin_ws/
  source devel/setup.bash
  roslaunch bwi_launch openai_gym.launch(This will start simulation environment in Gazebo)
  
  For training the agents : 
  rosrun bwi_segbot_hallway model_train.py
  
  For testing the agents using the trained model : 
  rosrun bwi_segbot_hallway model_test.py
