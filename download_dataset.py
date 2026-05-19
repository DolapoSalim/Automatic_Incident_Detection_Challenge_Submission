import os, zipfile, time, subprocess, sys

# Install gdown if needed
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
import gdown

# ── Target folder ──────────────────────────────────────────────────────────
BASE = r'C:\Users\DoLaPo\Documents\CODEBASE\Automatic_Incident_Detection_Challenge_Submission'

os.makedirs(os.path.join(BASE, 'data', 'train'), exist_ok=True)
os.makedirs(os.path.join(BASE, 'data', 'val'),   exist_ok=True)

# ── CSVs ───────────────────────────────────────────────────────────────────
print('Downloading train_GT.csv...')
gdown.download(id='1gktt4ZlS75ijx50ONbiFRkwWZkyb9b2l',
               output=os.path.join(BASE, 'data', 'train_GT.csv'), quiet=False)

print('Downloading val_GT.csv...')
gdown.download(id='1h1f4aMyMH845t5yrcllglXFSs5E_lX_2',
               output=os.path.join(BASE, 'data', 'val_GT.csv'), quiet=False)

# ── Train videos ───────────────────────────────────────────────────────────
print('\nDownloading train.zip...')
train_zip = os.path.join(BASE, 'data', 'train.zip')
t0 = time.time()
gdown.download(id='1-ffvPj8aGUUUnb4tzpp8nrPK4Kl-nHUF', output=train_zip, quiet=False)
print(f'Downloaded in {(time.time()-t0)/60:.1f} mins. Unzipping...')
with zipfile.ZipFile(train_zip, 'r') as z:
    z.extractall(os.path.join(BASE, 'data', 'train'))
os.remove(train_zip)
print(f'Train videos: {len(os.listdir(os.path.join(BASE, "data", "train")))} files')

# ── Val videos ─────────────────────────────────────────────────────────────
print('\nDownloading val.zip...')
val_zip = os.path.join(BASE, 'data', 'val.zip')
t0 = time.time()
gdown.download(id='1KOAbJg1yL5AnK9nMN8qesNJeCQX-O0Gc', output=val_zip, quiet=False)
print(f'Downloaded in {(time.time()-t0)/60:.1f} mins. Unzipping...')
with zipfile.ZipFile(val_zip, 'r') as z:
    z.extractall(os.path.join(BASE, 'data', 'val'))
os.remove(val_zip)
print(f'Val videos: {len(os.listdir(os.path.join(BASE, "data", "val")))} files')

print('\n Dataset ready!')
print(f'  train_GT.csv → {os.path.join(BASE, "data", "train_GT.csv")}')
print(f'  val_GT.csv   → {os.path.join(BASE, "data", "val_GT.csv")}')
print(f'  Train videos → {os.path.join(BASE, "data", "train")}')
print(f'  Val videos   → {os.path.join(BASE, "data", "val")}')