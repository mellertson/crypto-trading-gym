###################################################################################
# Bash functions to print a line and print a message surrounded by lines.
#
# Software Requirements:
#   - none
###################################################################################

print_line () {
	echo "--------------------------------------------------------------------------------"
}
print_msg () {
	print_line
	echo $1
	print_line
}

