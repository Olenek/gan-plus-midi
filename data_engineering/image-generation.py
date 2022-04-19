from midi2img import midi2image
import os

root_directory = os.fsencode('midi-files')
out_dir = os.fsencode('midi-images')

for root, dirs, _ in os.walk(root_directory):
    for subdir in dirs:
        print(subdir)
        for subroot, _, files in os.walk(os.path.join(root, subdir)):
            for file in files:
                path = os.path.join(subroot, file).decode('utf-8')
                out_path = os.path.join(out_dir, subdir).decode('utf-8')
                os.makedirs(out_path, exist_ok=True)
                midi2image(path, directory=out_path)


    # filename = os.fsdecode(file)
    # if filename.endswith(".asm") or filename.endswith(".py"):
    #     # print(os.path.join(directory, filename))
    #     continue
    # else:
    #     continue