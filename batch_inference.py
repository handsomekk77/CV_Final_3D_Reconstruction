import json
import os
import subprocess

# Settings
root_dir = r"D:/College_Documents/5th_grade_second_semester/Computer Vision/Final_Project/7SCENES_Original/7SCENES"
output_dir = r"D:\College_Documents\5th_grade_second_semester\Computer Vision\Final_Project\Results"
json_path = r"D:\College_Documents\5th_grade_second_semester\Computer Vision\Final_Project\reconstruction_sequences.json"
python_exe = "python"  # or "python3" depending on your system
script_path = r"D:\College_Documents\5th_grade_second_semester\Computer Vision\Final_Project\mast3r\3d_reconstruction.py"  # <-- change to your script name

# Load sequence list
with open(json_path, "r") as f:
    sequences = json.load(f)["sequences"]

os.makedirs(output_dir, exist_ok=True)

for seq_rel in sequences:
    seq_abs = os.path.join(root_dir, seq_rel)
    seq_name = os.path.basename(seq_rel)  # e.g., "seq-03"
    scene_folder = os.path.basename(os.path.dirname(seq_abs))  # e.g., "chess", "fire", etc.
    scene_name = f"{scene_folder}-{seq_name}"
    ply_path = os.path.join(output_dir, f"{scene_name}.ply")
    print(f"Processing {seq_abs} -> {ply_path}")

    # Call your script as a subprocess for each sequence
    cmd = [
        python_exe,
        script_path,
        "--dataset_path", seq_abs,
        "--output_dir", output_dir,
        "--scene_name", scene_name,
        "--save_ply"
        # ... add other args as needed ...
    ]
    subprocess.run(cmd, check=True)
