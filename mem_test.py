import os, sys, psutil, time

def print_memory():
    process = psutil.Process()
    print(f"[{time.strftime('%H:%M:%S')}] Memory (MB): {process.memory_info().rss / 1024 / 1024:.2f}")

print("Before import:")
print_memory()

from hr_analysis import run_hr_video_analysis
print("After import:")
print_memory()

# write a dummy 3 sec mp4
os.system(f"ffmpeg -y -f lavfi -i testsrc=duration=3:size=640x480:rate=30 dummy.mp4 -loglevel quiet")
print("After ffmpeg:")
print_memory()

try:
    res = run_hr_video_analysis("dummy.mp4")
    print(res)
except Exception as e:
    print(e)
finally:
    print("After processing:")
    print_memory()
    if os.path.exists("dummy.mp4"):
        os.remove("dummy.mp4")
