#!/bin/bash

# Get the current user's username
CURRENT_USER=$(whoami)

echo "Hi, $CURRENT_USER"

# Get the PIDs of vscode-server processes associated with the current user
PIDS=$(ps -elf | grep naravich | grep .vscode-server | awk '{print $4}')
PIDS=$(ps -elf | grep naravich | grep ipykernel | awk '{print $4}')

for PID in $PIDS; do
	echo "Killing vscode-server process with PID: $PID"
	kill $PID
done
