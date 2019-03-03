import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone as source:
    print("say something now")
    audio = r.listen(source)
    print("cool thanks")

try:
    print("TEXT:", r.recognize_google(audio))
except:
    pass
