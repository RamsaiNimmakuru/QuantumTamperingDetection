import yaml
from src.trainer import fit
cfg = yaml.safe_load(open("config/config_baseline.yaml"))
cfg['lr'] = float(cfg['lr']) # Convert learning rate to float
results = fit(cfg)
print("\n📊 FINAL METRICS")
for k,v in results.items():
    print(f"{k}: {v:.4f}")