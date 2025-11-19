import pandas as pd

df_client = pd.read_csv("datasets/client_metrics_uap_5min.csv")
df_cpu = pd.read_csv("datasets/cpu_metrics_uap_5min.csv")
df_memory = pd.read_csv("datasets/memory_metrics_uap_5min.csv")
df_5g = pd.read_csv("datasets/signal_5g_metrics_uap_5min.csv")
df_24g = pd.read_csv("datasets/signal_24g_metrics_uap_5min.csv")

df_client_avg = df_client.groupby("ap_name", as_index=False)["client_count"].mean()
df_client_avg.columns = ["ap_name", "avg_client_count"]

df_cpu_avg = df_cpu.groupby("ap_name", as_index=False)["cpu_usage_ratio"].mean()
df_cpu_avg.columns = ["ap_name", "avg_cpu_usage_ratio"]

df_memory_avg = df_memory.groupby("ap_name", as_index=False)[
    "memory_usage_ratio"
].mean()
df_memory_avg.columns = ["ap_name", "avg_memory_usage_ratio"]

df_5g_avg = df_5g.groupby("ap_name", as_index=False)["signal_dbm"].mean()
df_5g_avg.columns = ["ap_name", "avg_signal_5g_dbm"]

df_24g_avg = df_24g.groupby("ap_name", as_index=False)["signal_dbm"].mean()
df_24g_avg.columns = ["ap_name", "avg_signal_24g_dbm"]

merged_df_p1 = pd.merge(df_client_avg, df_cpu_avg, on="ap_name", how="outer")
merged_df_p2 = pd.merge(merged_df_p1, df_memory_avg, on="ap_name", how="outer")
merged_df_p3 = pd.merge(merged_df_p2, df_5g_avg, on="ap_name", how="outer")
merged_df_final = pd.merge(merged_df_p3, df_24g_avg, on="ap_name", how="outer")

merged_df_final.to_csv("average_metrics_uap_5min.csv", index=False)
