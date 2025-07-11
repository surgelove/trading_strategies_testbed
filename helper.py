import subprocess
import threading
from tkinter import messagebox
import pyttsx3

def say_nonblocking(text, voice=None):
    """
    Say text on macOS using the 'say' command in a non-blocking way
    
    Parameters:
    -----------
    text : str
        Text to speak
    voice : str, optional
        Voice to use (e.g., 'Alex', 'Samantha', 'Victoria')
    """
    print("Speaking:", text)
    def speak():
        try:
            cmd = ['say']
            if voice:
                cmd.extend(['-v', voice])
            cmd.append(text)
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Error with text-to-speech: {e}")
    
    # Run in a separate thread to avoid blocking
    thread = threading.Thread(target=speak, daemon=True)
    thread.start()

def say_hello():
    print("Hello")
    messagebox.showinfo("Greeting", "Hello")
    engine = pyttsx3.init()
    engine.say("Hello")
    engine.runAndWait()

