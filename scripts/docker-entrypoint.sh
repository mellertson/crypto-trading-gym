#!/bin/bash
###################################################################################
# Our standard Docker entrypoint for all Cybertron Ninja Docker services.
#
# This entrypoint will do the following:
#	- Retain all ENVs set in a service's Dockerfile and docker-compose.yaml file.
# 	- Source environment variables from Docker secrets, which are post-fixed
#	  with _FILE.  For example, and ENV named "MY_ENV_FILE" will read the contents
#  	  of the file '/run/secrets/MY_ENV_FILE' into an ENV named "MY_ENV".
#	- Run the Docker container's entrypoint command, which is specified in its
# 	  correspoding Dockerfile with the command: "CMD".
# 	  See: https://docs.docker.com/engine/reference/builder/#cmd for more info.
#
# Software Requirements:
#   - Bash script: `print-functions.sh`
#   - Bash script: `envs-from-secrets.sh`
###################################################################################

set -e
CWD="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. $CWD/print-functions.sh

print_line
echo "Executing ${CWD}/docker-entrypoint.sh ..."

# source ENVs from docker secrets.
. $CWD/envs-from-secrets.sh

# use ONLY in non-production containers!
if [[ $VERBOSE == "true" ]]; then
	print_line ""
	echo "ENVs:"
	env | sort
	print_line ""
fi

# start the app service
print_line ""
echo "Starting up container with the following command: "
echo "${@}"
exec "$@"
print_line ""





