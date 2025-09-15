import os

import pandas as pd

file = os.path.join("output", "synthetic_logs_csv", "event_log_rare_16000.csv")

output_file = file.replace(".csv", "_trim_ms.csv")

df = pd.read_csv(file, sep=';')

df['start_timestamp'] = df['start_timestamp'].str[:23]
df['end_timestamp'] = df['end_timestamp'].str[:23]

df.to_csv(output_file, index=False, sep=';')