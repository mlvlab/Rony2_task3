# python3 extract_frames.py --source video.mp4 --destination './' --start 1 --end 10000 --step 1000 
import os 
import shutil
import cv2
import argparse

def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    Output: video frames ill be written in the snapshot_frames directory created in frames_dir.
    """

    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(os.path.join(frames_dir,'snapshot_frames'))
    vname = video_path.split('/')[-1]
    print(frames_dir)
    if not os.path.isdir(frames_dir):
      os.mkdir(frames_dir)
    #else:
    #  filelist = glob.glob(os.path.join(frames_dir, "*.jpg"))
    #  for f in filelist:
    #    os.remove(f)

    video_dir, video_filename = os.path.split(video_path)
    assert os.path.exists(video_path)
    
    capture = cv2.VideoCapture(video_path)

    if start < 0:  
        start = 0
    if end < 0:
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start          # keep track of which frame we are up to, starting from start
    while_safety = 0       # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0        # a count of how many frames we have saved

    while frame < end:  

        _, image = capture.read()
        if while_safety > 500:
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:
            while_safety += 1
            continue

        if frame % every == 0:
            while_safety = 0  
            save_path = os.path.join(frames_dir, vname.split('.')[0]+"_{:010d}.jpg".format(frame))
            print(save_path)
            if not os.path.exists(save_path) or overwrite:
                cv2.imwrite(save_path, image) 
                saved_count += 1  

        frame += 1

    capture.release()

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='../../video_examples/election_2018_sample_1.mp4', help='source')
parser.add_argument('--destination', type=str, default='./', help='where to save')
parser.add_argument('--start', type=int, default=1, help='start frame')
parser.add_argument('--end', type=int, default=10000, help='end frame')
parser.add_argument('--step', type=int, default=1000, help='step frame')
opt = parser.parse_args()

# Run the snapshot extractor
extract_frames(opt.source, opt.destination, start=opt.start, end=opt.end, every=opt.step)
