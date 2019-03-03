from gtts import gTTS
import os     #will be on the top

tts = gTTS(text="Hey, you are Raghav", lang='en')
tts.save("audio.mp3")
os.system('mpg321 audio.mp3 -quiet')
