# MazeFollower


![](https://github.com/kunaljere/MazeFollower/blob/main/Maze_Demo.gif)


ROS 2 program that navigates a TurtleBot3 Burger through a maze by following directional signs until it reaches the goal (designated by the red target sign) in the shortest path possible. 

Nine different signs will be present in the maze, organized into four categories: wrong way (stop and do-no-enter signs) indicating the rob ot should turn around, goal, turn 90 degrees to the left (three left arrow signs), and turn 90 degrees to the right (three right arrow signs). The signs are colored and taped to the walls of the robot space.

labels - 0: empty wall, 1: left, 2: right, 3: do not enter, 4: stop, 5: goal.

![image](https://user-images.githubusercontent.com/43733660/206616995-34c21841-b218-4235-8e93-52f79704830c.png)

All relevant nodes are under the lilhero6_final folder

**getLidarData.py** - Receives lidar data and publishes the distance (in meters) of an object detected at angle 0.

**classifyImage.py** - Trains a knn model to classify images of signs and predicts a label (between 0 and 5) when the robot faces a wall.

**maze_runner.py** - Implements the functionality to navigate the maze using odometry, lidar distance, and image labels.
