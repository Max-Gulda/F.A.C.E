from gtts import gTTS
from tempfile import NamedTemporaryFile
from playsound import playsound
from datetime import datetime
import cv2 as cv
import time
import os
import threading


cooldowns = {"default": 0}

def play_sound(voice_path):
    try:
        playsound(voice_path, block=False)
        os.remove(voice_path)
        return True
    except Exception as e:
        print(f"Error while playing sound {e}")
        return False


def greeting_and_reminder(user, meetingTime=None, lang='sv', now=None):
    # Check if the user is currently on cooldown
    #if user in cooldowns and cooldowns[user] > time.time():
    #    # Calculate the amount of time left on the cooldown
    #    cooldown_left = round(cooldowns[user] - time.time(), 2)
    #    print(f"{user} is still on cooldown for {cooldown_left} seconds!")
    #    return

    # Get the current time if not provided
    now = now or datetime.now()
    hour, minute = int(now.strftime("%H")), int(now.strftime("%M"))  # Extract the current hour and minute

    daytime = ""
    # Determine the time of day based on the current hour
    if hour in range(4, 10):
        daytime = "morgon"
    elif hour in range(10, 17):
        daytime = "middag"
    elif hour in range(17, 24) or hour in range(0, 4):
        daytime = "kväll"
    namn_split = user.split("_")
    namn_join = " ".join(namn_split)
    # Generate the greeting text based on whether or not a meeting time was provided
    if meetingTime and meetingTime.strip():
        txt = f"God {daytime} {namn_join}, ditt nästa möte börjar klockan {meetingTime}"
    else:
        txt = f"God {daytime} {namn_join}"

    # Generate the spoken audio from the greeting text and play it
    try:
        gTTS(text=txt, lang=lang).write_to_fp(voice := NamedTemporaryFile())
        print(f"Temporary file path: {voice.name}")
        played = play_sound(voice.name)
        if not played:
            return
        print(f"Audio played for {user}")
    except Exception as e:
        print(f"Error while generating or playing audio: {e}")
        return

    # Add the user to the cooldown list with a 60 second cooldown period
    # Add 60 seconds to the current time for the cooldown
    #cooldowns[user] = time.time() + 60





def fancyDraw(img, x, y, x1, y1, l=30, t=5, color=(255, 0, 255)):
    """
    Draws a fancy rectangle on an image with four corners and a specified color and thickness.

    :param img: The image on which to draw the rectangle.
    :param x: The x-coordinate of the top-left corner of the rectangle.
    :param y: The y-coordinate of the top-left corner of the rectangle.
    :param x1: The x-coordinate of the bottom-right corner of the rectangle.
    :param y1: The y-coordinate of the bottom-right corner of the rectangle.
    :param l: The length of the lines drawn at each corner of the rectangle (default 30 pixels).
    :param t: The thickness of the lines drawn (default 5 pixels).
    :param color: The color of the lines and rectangle (default pink).

    :return: The image with the drawn rectangle.
    """
    
    # Draw the top-left corner of the rectangle
    cv.line(img, (x, y), (x + l, y), color, t)  # Draw a line from (x, y) to (x + l, y)
    cv.line(img, (x, y), (x, y + l), color, t)  # Draw a line from (x, y) to (x, y + l)

    # Draw the top-right corner of the rectangle
    cv.line(img, (x1, y), (x1 - l, y), color, t)  # Draw a line from (x1, y) to (x1 - l, y)
    cv.line(img, (x1, y), (x1, y + l), color, t)  # Draw a line from (x1, y) to (x1, y + l)

    # Draw the bottom-left corner of the rectangle
    cv.line(img, (x, y1), (x + l, y1), color, t)  # Draw a line from (x, y1) to (x + l, y1)
    cv.line(img, (x, y1), (x, y1 - l), color, t)  # Draw a line from (x, y1) to (x, y1 - l)

    # Draw the bottom-right corner of the rectangle
    cv.line(img, (x1, y1), (x1 - l, y1), color, t)  # Draw a line from (x1, y1) to (x1 - l, y1)
    cv.line(img, (x1, y1), (x1, y1 - l), color, t)  # Draw a line from (x1, y1) to (x1, y1 - l)

    # Draw the rectangle using the specified color and thickness
    cv.rectangle(img, (x, y), (x1, y1), color, 1)

    return img

if __name__ == "__main__":
    pass
