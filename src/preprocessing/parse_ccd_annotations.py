import ast

def parse_crash_1500(txt_path):
    accident_dict = {}

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",", 1)

            video_id = parts[0]

            # Extract label list safely
            label_str = parts[1].split("]")[0] + "]"
            frame_labels = ast.literal_eval(label_str)

            # First accident frame
            if 1 in frame_labels:
                accident_frame = frame_labels.index(1)
            else:
                accident_frame = None

            accident_dict[video_id] = accident_frame

    return accident_dict

