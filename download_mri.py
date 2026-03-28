from nilearn import datasets
import shutil, os

print('Downloading MNI152 brain MRI...')
data = datasets.fetch_icbm152_2009()
src = data.t1
print(f'Downloaded to: {src}')

os.makedirs('dataset', exist_ok=True)
shutil.copy(src, 'dataset/brain.nii.gz')
print('Saved to backend/dataset/brain.nii.gz')
