from scipy.ndimage import *

# Optuna + pLotting and visualising results
import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_slice, plot_contour
import plotly.io as pi

def normalize_to(batch, min_batch, max_batch, min_val=0.0, max_val=1.0, val=1.0, eps=1e-8):
#         if batch.ndim == 3:
#             batch = batch.unsqueeze(1)  # [B, 1, H, W]

#         # Compute batch-level min and max across all pixels in all images
#         min_batch = batch.min()
#         max_batch = batch.max()

        # Normalize the entire batch to [0, 1]
        normed = (batch - min_batch) / (max_batch - min_batch + eps)

        # Scale to [min_val, max_val]
        normed = normed * (max_val - min_val) + min_val

        return normed

def clbk(Study, trial):  
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv('optuna_results_short_full.csv', index=False)  # Save to csv file
    if trial.number%5==0:
        obj=plot_optimization_history(study)

        # plt.savefig('OptHist.png')

        pi.write_image(obj, 'OptHistnw1.png')

        obj1=plot_slice(study)
        pi.write_image(obj1, 'OpSlicenw1.png')

        obj2=plot_contour(study)
        pi.write_image(obj2, 'OptContnw1.png')

class resize(object):
    def __call__(self, sample):
        phase, image = sample['phase'], sample['image']
        
        # image[0] = minmax(np.sqrt(image[0]))
        # image[1] = minmax(np.sqrt(image[1]))
        
        phase = scipy.ndimage.zoom(phase, 256/np.shape(phase)[0], order=0)

        return {'phase': phase, 'image': image}
    

