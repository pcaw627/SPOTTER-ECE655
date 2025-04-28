import tkinter as tk
import requests
import pygame
import time
import os

URL_FEEDBACK = "http://3.15.203.82/json"
URL_IMAGE = "http://3.15.203.82/files/screenshot.png" 

previous_audio_file = "" 

def play_sound_from_string(name_string):
    filename = f"{name_string}"
    
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
    
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except pygame.error as e:
        print(f"An error occurred while trying to play the sound: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def fetch_and_display_feedback(feedback_label, image_label):
    global previous_audio_file 

    try:
        response = requests.get(URL_FEEDBACK)
        response.raise_for_status()
        data = response.json()

        feedback = data.get('feedback', 'No feedback found')
        feedback_label.config(text=feedback)

        feedback_audio_file = data.get('feedback_audio_file', '')

        # Only play the audio if the file URL has changed
        if feedback_audio_file and feedback_audio_file != previous_audio_file:
            
            play_sound_from_string(feedback_audio_file)
            previous_audio_file = feedback_audio_file

        img_response = requests.get(URL_IMAGE)
        img_response.raise_for_status()
        img_data = img_response.content

        img = tk.PhotoImage(data=img_data)
        img_width = img.width()
        img_height = img.height()

        max_width = 400
        max_height = 400

        width_scale = max_width / img_width
        height_scale = max_height / img_height

        scale = min(width_scale, height_scale)

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        img_resized = img.subsample(int(img_width / new_width), int(img_height / new_height))

        image_label.config(image=img_resized)
        image_label.image = img_resized

    except requests.RequestException as e:
        feedback_label.config(text=f"Network error:\n{e}")
    except ValueError as e:
        feedback_label.config(text=f"Data error:\n{e}")
    except Exception as e:
        feedback_label.config(text=f"Unexpected error:\n{e}")

    # Fetch every 2 seconds
    window.after(2000, fetch_and_display_feedback, feedback_label, image_label)  

def main():
    global window  
    window = tk.Tk()
    window.title("Pose Helper")
    window.geometry("1000x1000")

    feedback_label = tk.Label(window, text="Fetching feedback...", wraplength=400, font=("Arial", 14))
    feedback_label.pack(pady=20)

    image_label = tk.Label(window)
    image_label.pack(pady=20)

    fetch_and_display_feedback(feedback_label, image_label)

    window.mainloop()

if __name__ == "__main__":
    main()
