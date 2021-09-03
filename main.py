''' Evaluates the effect of the condition number of the linear operator of COMMIT on the fit.
Multiple different density maps were created by adding random noise. Then, the fit was performed
keeping the same tractogram and chaging the density map to fit to.
'''
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.segment.clustering import QuickBundlesX
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.metric import ResampleFeature
from dipy.tracking.streamline import set_number_of_points
import numpy as np
# import connectome_utils
import nibabel
import os
#import clustering_functions
import commit
from commit import trk2dictionary
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_orig()

#%% Setup experiments
cluster_thr     = 0.80

blur_sigma      = 0.4
blur_r_step     = 0.25
blur_gauss_min  = 0.1

phantom_path = '/home/nicola/Scrivania/ISBI2013'

#====================================================================

def get_streamlines_close_to_centroids( clusters, streamlines, n_pts ):
    """Return the streamlines closer to the centroids of each cluster.

    As first step, the streamlines of the input tractogram are resampled to n_pts points.
    """
    sample_streamlines = set_number_of_points(streamlines, n_pts)

    centroids_out = []
    for cluster in clusters:
        minDis      = 1e10
        minDis_idx  = -1
        centroid_fw = cluster.centroid
        centroid_bw = cluster.centroid[::-1]
        for i in cluster.indices:
            d1 = np.linalg.norm( centroid_fw - sample_streamlines[i] )
            d2 = np.linalg.norm( centroid_bw - sample_streamlines[i] )
            if d1>d2:
                dm = d2
            else:
                dm = d1

            if dm < minDis:
                minDis = dm
                minDis_idx = i
        centroids_out.append( streamlines[minDis_idx] )

    return centroids_out


def tractogram_cluster( filename_in, filename_reference, filename_out, thresholds, n_pts=20, random=True, verbose=False ) :
    """ Cluster streamlines in a tractogram.
    """
    if verbose :
        print( f'-> Clustering "{filename_in}":' )

    tractogram = load_tractogram( filename_in, reference=filename_reference, bbox_valid_check=False )
    if verbose :
        print( f'- {len(tractogram.streamlines)} streamlines found' )

    if np.isscalar( thresholds ) :
        thresholds = [ thresholds ]

    metric   = AveragePointwiseEuclideanMetric( ResampleFeature( nb_points=n_pts ) )

    if verbose :
        print( '- Running QuickBundlesX...' )
    if random == False :
        clusters = QuickBundlesX( thresholds, metric ).cluster( tractogram.streamlines )
    else:
        rng = np.random.RandomState()
        ordering = np.arange(len(tractogram.streamlines))
        rng.shuffle(ordering)
        clusters = QuickBundlesX( thresholds, metric ).cluster( tractogram.streamlines, ordering=ordering )
    if verbose :
        print( f'  * {len(clusters.leaves)} clusters in lowest level'  )

    if verbose :
        print( '- Replace centroids with closest streamline in input tractogram' )
    centroids = get_streamlines_close_to_centroids( clusters.leaves, tractogram.streamlines, n_pts )
    if verbose :
        print( f'  * {len(centroids)} centroids' )

    if verbose :
        print( f'- Save to "{filename_out}"' )
    tractogram_new = StatefulTractogram.from_sft( centroids, tractogram )
    save_tractogram( tractogram_new, filename_out, bbox_valid_check=False )


#====================================================================
'''
#%% Fit with COMMIT on original tractogram
#==========================================
# create dictionary (once for all experiments)
trk2dictionary.run(
    filename_tractogram = 'res_shifted_10.tck',
    filename_mask       = f'{phantom_path}/wm.nii.gz',
    fiber_shift         = 0.5,
    min_fiber_len       = 0,
    min_seg_len         = 1e-3,
    ndirs               = 1,
    path_out            = 'COMMIT_on_raw_tractogram'
)

# fit with COMMIT

mit = commit.Evaluation( '', '' )
mit.set_config('doNormalizeSignal', False)
mit.set_config('doMergeB0', False)
mit.set_config('doNormalizeKernels', True)

mit.load_data( f'{phantom_path}/ground-truth-density.nii.gz', f'{phantom_path}/ground-truth-density.scheme' )
mit.set_model( 'VolumeFractions' )
mit.model.set( hasISO=False )

mit.generate_kernels( regenerate=False, ndirs=1 )
mit.load_kernels()

mit.load_dictionary( 'COMMIT_on_raw_tractogram' )

mit.set_threads()
mit.build_operator()

mit.fit( tol_fun=1e-4, max_iter=2500, verbose=False )
mit.save_results()

# compute connectome
results_path = f'COMMIT_on_raw_tractogram/Results_VolumeFractions'
os.system( f'tck2connectome -force -quiet -symmetric -assignment_radial_search 2 -out_assignments {results_path}/fibers_assignment.txt -tck_weights_in {results_path}/streamline_weights.txt res_shifted_10.tck {phantom_path}/gm.nii.gz {results_path}/fibers_connectome.csv')

'''
#%% Fit with COMMIT_blur on raw
#=================================================

# perform clustering
#tractogram_cluster('res_shifted_2.tck', f'{phantom_path}/wm.nii.gz', f'fibers_connecting_clustered_th={cluster_thr:.1f}.tck', [10.0, cluster_thr], random=False, verbose=True)


"""
# create dictionary (once for all experiments)
r_max = np.sqrt( -2.0 * blur_sigma**2 * np.log( blur_gauss_min ) )
r_max_round = np.ceil(r_max/blur_r_step)*blur_r_step
blur_radii   = np.arange( blur_r_step, r_max_round+1e-6, blur_r_step )
blur_samples = 4*np.arange(1,len(blur_radii)+1)

trk2dictionary.run(
    filename_tractogram = f'fibers_connecting_clustered_th={cluster_thr:.1f}.tck',
    filename_mask       = f'{phantom_path}/wm.nii.gz',
    fiber_shift         = 0.5,
    min_fiber_len       = 0,
    min_seg_len         = 1e-3,
    ndirs               = 1,
    blur_sigma          = blur_sigma,
    blur_radii          = blur_radii,
    blur_samples        = blur_samples,
    path_out            = 'COMMIT_on_clustered_tractogram'
)

# fit with COMMIT

mit = commit.Evaluation( '', '.' )
mit.set_config('doNormalizeSignal', False)
mit.set_config('doMergeB0', False)
mit.set_config('doNormalizeKernels', True)

mit.load_data( f'{phantom_path}/ground-truth-density.nii.gz', f'{phantom_path}/ground-truth-density.scheme' )
mit.set_model( 'VolumeFractions' )
mit.model.set( hasISO=False )

mit.generate_kernels( regenerate=False, ndirs=1 )
mit.load_kernels()

mit.load_dictionary( 'COMMIT_on_clustered_tractogram' )

mit.set_threads()
mit.build_operator()

mit.fit( tol_fun=1e-4, max_iter=2500, verbose=False )
mit.save_results()

# compute connectome
tractoram_filename = f'fibers_connecting_clustered_th={cluster_thr:.1f}.tck'
results_path = f'COMMIT_on_clustered_tractogram/Results_VolumeFractions'
os.system( f'tck2connectome -force -quiet -symmetric -assignment_radial_search 2 -out_assignments {results_path}/fibers_assignment.txt -tck_weights_in {results_path}/streamline_weights.txt {tractoram_filename} {phantom_path}/gm.nii.gz {results_path}/fibers_connectome.csv')

"""

# create dictionary (once for all experiments)
r_max = np.sqrt( -2.0 * blur_sigma**2 * np.log( blur_gauss_min ) )
r_max_round = np.ceil(r_max/blur_r_step)*blur_r_step
blur_radii   = np.arange( blur_r_step, r_max_round+1e-6, blur_r_step )
blur_samples = 4*np.arange(1,len(blur_radii)+1)

trk2dictionary.run(
    filename_tractogram = f'res_shifted_10.tck',
    filename_mask       = f'{phantom_path}/wm.nii.gz',
    fiber_shift         = 0.5,
    min_fiber_len       = 0,
    min_seg_len         = 1e-3,
    ndirs               = 1,
    blur_sigma          = blur_sigma,
    blur_radii          = blur_radii,
    blur_samples        = blur_samples,
    path_out            = 'COMMIT_on_clustered_tractogram'
)

# fit with COMMIT

mit = commit.Evaluation( '', '.' )
mit.set_config('doNormalizeSignal', False)
mit.set_config('doMergeB0', False)
mit.set_config('doNormalizeKernels', True)

mit.load_data( f'{phantom_path}/ground-truth-density.nii.gz', f'{phantom_path}/ground-truth-density.scheme' )
mit.set_model( 'VolumeFractions' )
mit.model.set( hasISO=False )

mit.generate_kernels( regenerate=False, ndirs=1 )
mit.load_kernels()

mit.load_dictionary( 'COMMIT_on_clustered_tractogram' )

mit.set_threads()
mit.build_operator()

mit.fit( tol_fun=1e-4, max_iter=2500, verbose=False )
mit.save_results()

# compute connectome
tractoram_filename = 'res_shifted_10.tck'
results_path = f'COMMIT_on_clustered_tractogram/Results_VolumeFractions'
os.system( f'tck2connectome -force -quiet -symmetric -assignment_radial_search 2 -out_assignments {results_path}/fibers_assignment.txt -tck_weights_in {results_path}/streamline_weights.txt {tractoram_filename} {phantom_path}/gm.nii.gz {results_path}/fibers_connectome.csv')


