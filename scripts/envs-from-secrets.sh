#!/bin/bash
###################################################################################
# Converts Docker secrets, which are mounted as files, into ENV variables
# for all file names ending in "_FILE".
#
# docker-compose.yaml example:
# 	version: "3.8"
#	secrets:
#   	SERVICE_DB_PASSWORD:
#       	external: true
#	services:
#		api:
#    		secrets:
#    		    -   source: SERVICE_DB_PASSWORD
#    		        target: POSTGRES_PASSWORD
#					# ^^ will mount secret in file: /run/secrets/POSTGRES_PASSWORD
#	 		environment:
#        		POSTGRES_PASSWORD_FILE: /run/secrets/POSTGRES_PASSWORD
#				# ^^ will create variable POSTGRES_PASSWORD set to value of secret SERVICE_DB_PASSWORD
#
# Software Requirements:
# 	- Bash script: `print-functions.sh`
###################################################################################
set -e
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. $SCRIPTS_DIR/print-functions.sh

source_envs_from_secrets () {
  print_line
	regex="^(\w+)(_FILE)\=(.*)$"
  for VARIABLE in $(env); do
    if [[ $VARIABLE =~ $regex ]]; then
      # match
      VARIABLE_NAME="${BASH_REMATCH[1]}"
      SECRET_FILENAME="${BASH_REMATCH[3]}"
      echo "exporting ENV ${VARIABLE_NAME} from value contained in file ${SECRET_FILENAME}"
      export "${VARIABLE_NAME}"="$(cat ${SECRET_FILENAME})"
    fi
  done
  print_line
}

# source ENVs from Docker secrets
source_envs_from_secrets



