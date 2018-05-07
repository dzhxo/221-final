import sys
import moviepy.video.io.ffmpeg_tools as movie_tools
import os

TASK_NAMES = ['intro', 'scene', 'stress', 'break', 'necker', 'survey']

if len(sys.argv) < 3:
    print "Need more arguments"
    sys.exit()

OUTPUT_DIR_PATH = "spliced_data"
TIMES_CSV = open(sys.argv[1])
TIMES_CSV.readline()
for line in TIMES_CSV:
    vals = line.split(',')
    subject = vals[0]
    scene = vals[1]
    del vals[:2]
    for i in range(len(vals)):
        if not vals[i]:
            continue
        task_name = TASK_NAMES[i]
        video_name = "" # TODO: FIGURE OUT VIDEO NAMES
        #TODO: FIGURE OUT WHAT IF MISSING i+1 IN MIDDLE
        end = vals[i + 1] if len(vals) > i + 1 else None
        output_path = os.path.join(OUTPUT_DIR_PATH, subject + '_' + scene + '_' + task_name)
        load_and_splice_video(video_name, task_name, vals[i], end, output_path)

def load_and_splice_video(video_name, task, start, end, output):
    start = start.split(':')
    if len(start) > 2:
        del start[2]
    new_start = int(start[0]) * 60 + int(start[1])
    end = end.split(':')
    if len(end) > 2:
        del end[2]
    new_end = int(end[0]) * 60 + int(end[1])
    movie_tools.extract_subclip(video_name, new_start, new_end, output)
