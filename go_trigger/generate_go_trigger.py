# generate_go_trigger.py
import cv2
import numpy as np

def create_trigger_video(output_path='countdown_trigger.mp4', fps=30):
    width, height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    def add_frames(img, duration_sec):
        for _ in range(int(fps * duration_sec)):
            out.write(img)
    
    # black background
    black = np.zeros((height, width, 3), dtype=np.uint8)
    
    # countdown
    for num in ['3', '2', '1']:
        frame = black.copy()
        cv2.putText(frame, num, (width//2-200, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 20, (255,255,255), 15)
        add_frames(frame, 1.0)
    
    # WHITE FLASH (this is your trigger)
    white = np.full((height, width, 3), 255, dtype=np.uint8)
    add_frames(white, 0.1)  # 100ms flash
    
    # GO text
    frame = black.copy()
    cv2.putText(frame, 'GO', (width//2-300, height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 20, (0,255,0), 15)
    add_frames(frame, 2.0)  # hold for 2s
    
    out.release()
    print(f"✓ Trigger video saved: {output_path}")

if __name__ == '__main__':
    create_trigger_video()