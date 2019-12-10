# bwi_segbot_hallway

Do the initial setup of BWI project using this repository https://github.com/ub31/bwi
Pull bwi_segbot_hallway repository under bwi_common folder and do a catkin build.

After build for launching the training script :
  cd ~/catkin_ws/
  source devel/setup.bash
  roslaunch bwi_launch openai_gym.launch(This will start simulation environment in Gazebo)
  
  For training the agents : 
  rosrun bwi_segbot_hallway model_train.py
  For testing the agents using the trained model : 
  rosrun bwi_segbot_hallway model_test.py
