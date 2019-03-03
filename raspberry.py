import subprocess
import time
subprocess.call("""ssh pi@143.215.98.197 -t 'python ./Desktop/moveServo.py'""", shell=True)
