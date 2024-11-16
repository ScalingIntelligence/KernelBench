import os

dir = "/matx/u/aco/KernelBenchInternal/KernelBench/old_level3"
# start_index = len(os.listdir("./level1"))
start_index = 30

i = 1
files = os.listdir(dir)
files = sorted(files, key=lambda x: x.split("_")[1])
# print(files)
# raise
for file in files:
    # old name is X_some_other_words.py
    # new name is {start_index + i}_some_other_words.py
    new_name = f"{start_index + i}_{'_'.join(file.split('_')[1:])}"
    os.rename(f"{dir}/{file}", f"{dir}/{new_name}")
    i += 1
