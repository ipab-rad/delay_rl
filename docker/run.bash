#!/bin/bash
nvidia-docker run -ti --rm --entrypoint /bin/bash --name delay_sensor_simon \
			 --group-add sudo \
			 -v /home/artificialsimon/artificial/projects/rl_random_agents_time_delay/src/todor_mujoco_docker/docker-cuda-gym/tensorflow/mujoco:/root/.mujoco \
			 -v /home/artificialsimon/artificial/projects/rl_random_agents_time_delay:/root \
			 ta/gym 
#	 		 --user $(id -u):$(id -g) \
