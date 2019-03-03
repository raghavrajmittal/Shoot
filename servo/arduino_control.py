import serial
import time
import atexit

arduino = None

def rotate(arduino):
	arduino.write(("R\n").encode())

def connect(port = '/dev/cu.usbserial-14430'):
	global arduino
	arduino = serial.Serial(port, 9600, timeout=0.5)
	print(arduino)
	time.sleep(1)
	return arduino

def disconnect(arduino):
	arduino.close()
