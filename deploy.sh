#!/bin/bash

docker stack deploy -c docker-compose-deploy.yml analyzing-simulations

# for stopping the stack
# docker stack rm analyzing-simulations