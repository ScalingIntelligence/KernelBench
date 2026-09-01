import matplotlib.pyplot as plt
import pandas as pd

file_path = 'all_perf.csv'

df = pd.read_csv(file_path)

names = df['name'].unique()

for name in names:
    subset = df[df['name'] == name]
    plt.plot(subset['seqlen'], subset['bw'], label=name)

plt.title('bandwidth')
plt.xlabel('seqlen')
plt.ylabel('bw (GB/s)')
plt.legend()

plt.savefig('bandwidth_vs_seqlen.png')