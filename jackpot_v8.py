import time

from imutils import face_utils
from utils import *
from multiprocessing import Process
import pyttsx3
import datetime
import speech_recognition as sr
import wikipedia as w
import webbrowser
import smtplib
import numpy as np
import pyautogui as pag
import os
import imutils
import dlib
import cv2
import serial
import pyglet
import time
from pynput.mouse import Button, Controller
import whatsapp

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty("rate", 170)

mouse = Controller()

joysticks = pyglet.input.get_joysticks()

if joysticks:
    joystick = joysticks[0]

joystick.open()

is_moving_x, is_moving_y, is_scrolling = False, False, False
mouse_x, mouse_y, scroll_y = 0, 0, 0


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening...')
        r.pause_threshold =1
        r.energy_threshold = 300
        audio = r.listen(source,0,4)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except :
        print("\n Say again please")
        return "None"
    return query


def video():

   # ser = serial.Serial('com4',9600)
    MOUTH_AR_THRESH = 0.2
    MOUTH_AR_CONSECUTIVE_FRAMES = 15
    EYE_AR_THRESH = 0.19
    EYE_AR_CONSECUTIVE_FRAMES = 15
    WINK_AR_DIFF_THRESH = 0.04
    WINK_AR_CLOSE_THRESH = 0.01
    WINK_CONSECUTIVE_FRAMES = 3

    MOUTH_COUNTER = 0
    EYE_COUNTER = 0
    WINK_COUNTER = 0
    INPUT_MODE = False
    EYE_CLICK = False
    LEFT_WINK = False
    RIGHT_WINK = False
    SCROLL_MODE = False
    ANCHOR_POINT = (0, 0)
    WHITE_COLOR = (255, 255, 255)
    YELLOW_COLOR = (0, 255, 255)
    RED_COLOR = (0, 0, 255)
    GREEN_COLOR = (0, 255, 0)
    BLUE_COLOR = (255, 0, 0)
    BLACK_COLOR = (0, 0, 0)

    shape_predictor = "model/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    # Video capture
    vid = cv2.VideoCapture(0)
    resolution_w = 1366
    resolution_h = 768
    cam_w = 640
    cam_h = 480
    unit_w = resolution_w / cam_w
    unit_h = resolution_h / cam_h
    if SCROLL_MODE :
        speak('scroll mode activated')

    while True:
        #ser.write(bytearray('N', 'ascii'))
        _, frame = vid.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=cam_w, height=cam_h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        if len(rects) > 0:
            rect = rects[0]
        else:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            continue

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        mouth = shape[mStart:mEnd]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        nose = shape[nStart:nEnd]

        temp = leftEye
        leftEye = rightEye
        rightEye = temp

        mar = mouth_aspect_ratio(mouth)
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        diff_ear = np.abs(leftEAR - rightEAR)

        nose_point = (nose[3, 0], nose[3, 1])

        mouthHull = cv2.convexHull(mouth)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
        cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
        cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

        for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
            cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)


        if diff_ear > WINK_AR_DIFF_THRESH:

            if leftEAR < rightEAR:
                if leftEAR < EYE_AR_THRESH:
                    WINK_COUNTER += 1
                else :
                    WINK_COUNTER = 0

            elif leftEAR > rightEAR:
                if rightEAR < EYE_AR_THRESH:
                    WINK_COUNTER += 1
                else :
                    WINK_COUNTER =0
        else:
            if ear <= EYE_AR_THRESH:
                EYE_COUNTER += 1

                if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES:
                    SCROLL_MODE = not SCROLL_MODE
                    # INPUT_MODE = not INPUT_MODE
                    EYE_COUNTER = 0

                    # nose point to draw a bounding box around it

            else:
                EYE_COUNTER = 0
                WINK_COUNTER = 0

        if mar > MOUTH_AR_THRESH:
            MOUTH_COUNTER += 1

            if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
                # if the alarm is not on, turn it on
                INPUT_MODE = not INPUT_MODE
                # SCROLL_MODE = not SCROLL_MODE
                MOUTH_COUNTER = 0
                ANCHOR_POINT = nose_point

        else:
            MOUTH_COUNTER = 0

        if INPUT_MODE:

            #ser.write(bytearray('Y', 'ascii'))
            cv2.putText(frame, "READING INPUT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
            x, y = ANCHOR_POINT
            nx, ny = nose_point
            w, h = 40, 25
            multiple = 1
            cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
            cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

            dir = direction(nose_point, ANCHOR_POINT, w, h)
            cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
            drag = 17
            if dir == 'right':
                pag.moveRel(drag, 0)
            elif dir == 'left':
                pag.moveRel(-drag, 0)
            elif dir == 'up':
                if SCROLL_MODE:
                    pag.scroll(40)
                else:
                    pag.moveRel(0, -drag)
            elif dir == 'down':
                if SCROLL_MODE:
                    pag.scroll(-40)
                else:
                    pag.moveRel(0, drag)

        if SCROLL_MODE:
            cv2.putText(frame, 'SCROLL MODE IS ON!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)


        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            speak('Mouse mode deactivated')
            break

    # Do a bit of cleanup
    cv2.destroyAllWindows()
    vid.release()


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 4 and hour < 12:
        speak("Good Morning!")
    elif hour >= 12 and hour < 16:
        speak("Good afternoon")
    else:
        speak("Good Evening")
    speak("My Name is Jackpot!")
    speak('What can i do for you sir?')


def sendEmail(to, content):
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.ehlo()
    server.starttls()
    server.login('vikasdewangan336@gmail.com', 'mshkizwnoubypukd')
    speak('You have logged in')
    server.sendmail('vikasdewangan218@gmail.com', to, content)
    server.quit()

def respond(AudioString):
    pyttsx3.speak(AudioString)

def voice():

    while True:
        query = takeCommand().lower()

        if 'wikipedia' in query:
            speak('Searching wikipedia...')
            try:
                query = query.replace("wikipedia", "")
                results = w.summary(query, sentences=2)
                speak('you said' + query)
                speak('According to wikipedia')
                print(results)
                speak(results)
            except:
                speak('invalid input.try again sir!')
        elif 'yourself' in query:
            speak('you said' + query )
            speak('Here we go')
            speak('My name is Jackpot')
            speak('My parents are Vikas  Dewangan')
            speak('and Wasim  akhtar khan')
            speak('I was created to help those people who cannot use their hands properly and')
            speak('also helps those people who are lazy like me.')
        elif 'youtube' in query:
            speak('you said' + query)
            query = query.replace("youtube", "")
            speak("What Do I Search")
            q = takeCommand()
            web = "www.youtube.com/results?search_query=" + q
            webbrowser.open(web)
        elif 'google' in query:
            speak('you said' + query)
            speak("What do I search")
            q = takeCommand()
            try:
                pywhatkit.search(q)
                results = wikipedia.summary(q)
                speak(results)
            except:
                speak("not found")
        elif 'reddit' in query:
            speak('you said' + query)
            webbrowser.open("reddit.com")
        elif 'time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"The time is {strTime}")
        elif 'song' in query:
            speak('you said' + query)
            spot = "C:\\Users\\Administrator\\AppData\\Roaming\\Spotify\\Spotify.exe"
            os.startfile(spot)
        elif 'mail' in query:
            try:
                speak("What should I say?")
                content = takeCommand()
                to = "vikasdewangan218@gmail.com"
                sendEmail(to, content)
                speak("Email has been sented!")
            except Exception as e:
                print(e)
                speak("Sorry can't send the mail at this moment")
        elif 'left' in query:
            pag.click(button='left')
            speak('left click')
        elif 'right' in query:
            pag.click(button='right')
            speak('right click')
        elif 'jump' in query:
            pag.click(button='middle')
            speak('middle click')
        elif 'double' in query:
            pag.click(clicks=2)
            speak('double click')
        elif 'play' in query:
            pag.press('space')
        elif 'stop' in query:
            pag.press('space')
        elif 'close' in query:
            pag.hotkey('alt', 'f4')
        elif 'refresh' in query:
            pag.hotkey('ctrl', 'r')
        elif 'escape' in query:
            pag.press('esc')
        elif 'escape' in query:
            pag.press('esc')
        elif 'check' in query:
            speak("Opening gmail.com")
            webbrowser.open("gmail.com")

        elif 'exit' in query:
            speak('Have a great day sir!')
            speak('Goodbye')
            exit(0)

def on_joybutton_press(joystick, button):
    if button == 2:
        pag.click(button='left')  # Circle
    if button == 3:
        pag.click(button='right')  # Triangle
    if button == 1:
        pag.hotkey('alt', 'tab')  # Cross
    if button == 4:
        pag.press('space')  # L1

def on_joyhat_motion(joystick, hat_x, hat_y):
    print(hat_x, hat_y)


def on_joyaxis_motion(joystick, axis, value):
    global is_moving_x, is_moving_y, mouse_x, mouse_y, is_scrolling, scroll_y

    if axis == 'x':
        if abs(value) > 0.2:
            is_moving_x = True
            mouse_x = value * 6
        else:
            is_moving_x = False
            mouse_x = 0
    elif axis == 'y':
        if abs(value) > 0.2:
            is_moving_y = True
            mouse_y = value * 6
        else:
            is_moving_y = False
            mouse_y = 0
    elif axis == 'rz':
        if abs(value) > 0.2:
            is_scrolling = True
            scroll_y = -value
        else:
            is_scrolling = False
            scroll_y = 0


joystick.push_handlers(
    on_joybutton_press,
    on_joyhat_motion,
    on_joyaxis_motion
)


def mouse_move(dt):
    global is_moving_x, is_moving_y, mouse_x, mouse_y, is_scrolling, scroll_y

    if is_moving_x or is_moving_y:
        mouse.move(mouse_x, mouse_y)


def mouse_scroll(dt):
    if is_scrolling:
        mouse.scroll(0, scroll_y)


pyglet.clock.schedule_interval(mouse_move, interval=0.005)
pyglet.clock.schedule_interval(mouse_scroll, interval=0.1)

if __name__ == '__main__':
    wishMe()
    time.sleep(1)
    Process(target=video).start()
    Process(target=voice).start()
    speak('Joystick mouse control and voice control is now ready for use ')
    pyglet.app.run()
    speak('and the video capture will be start in four seconds')





