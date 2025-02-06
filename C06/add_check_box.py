import os

label_folder = 'results/part2/gen_labels'
checkbox_folder = 'labels/checkbox/MS04_Exam'

label_files = os.listdir(label_folder)
checkbox_files = os.listdir(checkbox_folder)
checkbox_file_name = [os.path.splitext(file)[0] for file in checkbox_files]

for label_file in label_files:
    label_file_path = os.path.join(label_folder, label_file)
    file_name = os.path.splitext(label_file)[0]
    if file_name[:-2] in checkbox_file_name:
        print(label_file_path)
        lines = list()
        with open(os.path.join(checkbox_folder, f"{file_name[:-2]}.txt"), 'r') as f:
            checkboxes = f.readlines()
        for i in range(len(checkboxes)):
            checkbox = checkboxes[i].strip().split(' ')
            checkbox[0] = str(int(checkbox[0]) + 1)
            lines.append(' '.join(checkbox) + '\n')
        with open(label_file_path, 'a') as f:
            f.write(''.join(lines))
        print(f'done {label_file_path}')