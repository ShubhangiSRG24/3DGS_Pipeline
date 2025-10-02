import cv2
import os
def extract_frames(video_path, num_frames, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: 영상 파일을 열 수 없음 {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"영상의 전체 프레임 개수: {total_frames}")
    
    interval = max(total_frames // num_frames, 1)
    print(f"Extracting {num_frames} frames at interval: {interval}")
    
    current_frame = 0
    extracted_frames = 0
    
    while cap.isOpened() and extracted_frames < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if current_frame % interval == 0:
            frame_filename = os.path.join(output_folder, f"{extracted_frames:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {extracted_frames} at {frame_filename}")
            extracted_frames += 1
        
        current_frame += 1
    
    cap.release()
    print(f"Extracted {extracted_frames} frames to {output_folder}")
if __name__ == "__main__":
    video_path = input("Where is your video? (video file path (ex: data/<Project Name>/<Project vid>.mp4): ")
    num_frames = int(input("How many images you need? (int): "))
    output_folder = input("Where do you want to save? (output file path (ex: data/<Project Name>/input): ")
    
    extract_frames(video_path, num_frames, output_folder)
