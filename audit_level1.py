import os
import difflib

original_dir = "KernelBench/level1"
metr_dir = "metr_tasks/level_1"

original_files = set(os.listdir(original_dir))
metr_files = set(os.listdir(metr_dir))

common = original_files & metr_files
only_in_original = original_files - metr_files
only_in_metr = metr_files - original_files

print("\n✅ IDENTICAL FILES:")
for filename in sorted(common):
    with open(os.path.join(original_dir, filename)) as f1, open(os.path.join(metr_dir, filename)) as f2:
        if f1.read() == f2.read():
            print(f"  {filename}")

print("\n✏️ MODIFIED FILES:")
for filename in sorted(common):
    with open(os.path.join(original_dir, filename)) as f1, open(os.path.join(metr_dir, filename)) as f2:
        if f1.read() != f2.read():
            print(f"  {filename}")

print("\n❌ MISSING FROM METR:")
for filename in sorted(only_in_original):
    print(f"  {filename}")

print("\n🆕 ONLY IN METR:")
for filename in sorted(only_in_metr):
    print(f"  {filename}")
