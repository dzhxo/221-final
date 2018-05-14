#pip install ffprobe

from ffprobe import FFProbe
import sys, os
import cv2

def time_to_ms(raw_time):
    split_time = raw_time.split(':')
    if len(split_time) >= 2:
        min = int(split_time[0])
        sec = int(split_time[1])
        return float(1000*(min*60 + sec))


TASK_NAMES = ['intro', 'scene', 'stress', 'break', 'necker', 'survey']

#if len(sys.argv) < 3:
#    print "Need more arguments"
#    sys.exit()

INTERVAL_BETWEEN_FRAMES = 500

OUTPUT_DIR_PATH = "spliced_data"
TIMES_CSV = open(sys.argv[1])
TIMES_CSV.readline()

for line in TIMES_CSV:
    vals = line.split(',')
    print(vals)
    subject = vals[0]
    scene = vals[1]
    print subject,scene
    filename = str(subject.strip()) + '.mov'
    if os.path.isfile(filename):
        cap = cv2.VideoCapture(filename)
        tot_len = int(float(FFProbe(filename).video[0].duration) * 1000)
        for i in range(len(TASK_NAMES)): 
            curr_time = vals[2+i].strip()
            if curr_time == '' or vals[3+i].strip() == '':
                continue

            start_time = 0
            end_time = 0

            if i < len(TASK_NAMES) - 1:
                start_time = time_to_ms(vals[2+i])
                print(vals[2+i])
                end_time = time_to_ms(vals[3+i])
                print(vals[3+i])
            else: # end
                start_time = time_to_ms(vals[2+i])
                end_time = tot_len

            curr_time = start_time
            frame_num = 0
            while curr_time < end_time:
                cap.set(cv2.CAP_PROP_POS_MSEC, curr_time)
                success,image = cap.read()
                if success:
                    small = cv2.resize(image, (0,0), fx=0.3, fy=0.3)
                    cv2.imwrite("frame"+str(frame_num)+"_"+str(subject)+"_"+scene[:3]+"_"+TASK_NAMES[i][:3]+".jpg", small)
                curr_time += INTERVAL_BETWEEN_FRAMES
                frame_num += 1
 