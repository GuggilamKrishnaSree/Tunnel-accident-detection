import subprocess

def normalize_video(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,

        # ---- VIDEO ----
        "-map", "0:v:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",

        # ---- AUDIO (IMPORTANT) ----
        "-map", "0:a?",          # include audio if present
        "-c:a", "aac",
        "-b:a", "128k",

        # ---- STREAMING FIX ----
        "-movflags", "+faststart",

        output_path
    ]

    subprocess.run(cmd, check=True)
