from os import system
import serial
import subprocess
from time import sleep

# Start PPPD
def open_pppd():
	# Check if PPPD is already running by looking at syslog output
	output1 = subprocess.check_output("cat /var/log/syslog | grep pppd | tail -1", shell=True)
	if "secondary DNS address" not in output1 and "locked" not in output1:
		while True:
			# Start the "fona" process
			subprocess.call("sudo pon fona", shell=True)
			sleep(2)
			output2 = subprocess.check_output("cat /var/log/syslog | grep pppd | tail -1", shell=True)
			if "script failed" not in output2:
				break
	# Make sure the connection is working
	while True:
		output2 = subprocess.check_output("cat /var/log/syslog | grep pppd | tail -1", shell=True)
		output3 = subprocess.check_output("cat /var/log/syslog | grep pppd | tail -3", shell=True)
		if "secondary DNS address" in output2 or "secondary DNS address" in output3:
			return True

# Stop PPPD
def close_pppd():
	print "turning off cell connection"
	# Stop the "fona" process
	subprocess.call("sudo poff fona", shell=True)
	# Make sure connection was actually terminated
	while True:
		output = subprocess.check_output("cat /var/log/syslog | grep pppd | tail -1", shell=True)
		if "Exit" in output:
			return True

def send_serial(msg):
	#TODO need to fix the serial number and check the other arguments
	print "sending message"
	#Start serial connection
	ser=serial.Serial('/dev/serialAIA', 115200, bytesize=serial.EIGHTBITS, parity.PARITY_NONE, stopbits.STOPBITS_ONE, timeout=1) 
	#send mgs TODO check the AT code for sending message
	ser.write("AT+1234\r" + msg)
	while True:
		responce = ser.readline()
		#TODO check what the expected response is to validate message was sent
		if "dunno" in response:
			print response
			return True
		#If the message was not sent try again
		if "alsodunno" in response:
			sleep(5)
			ser.write("AT+1234\r" + msg)
		else: 
			ser.write("AT+1234\r" + msg)


# Start the program by opening the cellular connection 
def send_message(msg):
	if open_pppd():
		while True:
			# Close the cellular connection
			if close_pppd():
				print "closing connection"
				sleep(1)
			#TODO figure out how to get message from faw_detector.py
			send_serial(msg)	
