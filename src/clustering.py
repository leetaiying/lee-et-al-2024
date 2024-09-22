import logging
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Callable, Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tqdm.notebook import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M') # %Y-%m-%d

def get_n_retained_pc(pca: PCA) -> int:
    """Determine the number of principal components (PCs) to retain based on the scree plot elbow method.

    This function calculates the number of PCs to keep by identifying the "elbow" in the scree plot, where the
    explained variance ratio significantly decreases. The method follows the implementation used by Namboodiri
    et al. (2019).

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        An instance of sklearn's PCA class, fitted to the data.

    Returns
    -------
    int
        The number of principal components to retain for downstream analysis.
    """
    pca_vectors = pca.components_
    print(f'Total number of PCs: {pca_vectors.shape[0]}')

    explained_var_ratio = pca.explained_variance_ratio_
    residuals = explained_var_ratio - (
        explained_var_ratio[0] + # variance explained by the first PC
        (explained_var_ratio[-1] - explained_var_ratio[0]) / (explained_var_ratio.size - 1) * np.arange(explained_var_ratio.size)
        )
    num_retained_pcs = int(np.argmin(residuals))
    print(f'Number of PCs to keep: {num_retained_pcs}')

    return num_retained_pcs

def scree_plot(
        pca: PCA,
        num_retained_pcs: int,
        n_pc_displayed: int = 100
        ) -> Tuple[Figure, Axes]:
    """
    Generate a scree plot to visualize the fraction of variance explained by each principal component.

    This function creates a scree plot that displays the explained variance ratio for each principal component (PC)
    derived from a PCA analysis. It also highlights the pre-determined number of components to retain.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        An instance of sklearn's PCA class, fitted to the data.
    num_retained_pcs : int
        The number of principal components to retain, as determined by the elbow method in `get_n_retained_pc()` or other criterion.
    n_pc_displayed : int, optional
        The maximum number of principal components to display on the scree plot. Default is 100.

    Returns
    -------
    Tuple[Figure, Axes]
        A tuple containing:
        - `fig`: The matplotlib `Figure` object of the scree plot.
        - `ax`: The matplotlib `Axes` object of the scree plot.

    """
    fig, ax = plt.subplots(figsize=(3,3))
    plt.rc('font', size=14)
    explained_var_ratio = pca.explained_variance_ratio_
    ax.plot(np.arange(explained_var_ratio.shape[0]).astype(int)+1, explained_var_ratio, 'k')
    ax.set_ylabel('Fraction of\nvariance explained', fontsize=14)
    ax.set_xlabel('PC number', fontsize=14)
    ax.axvline(num_retained_pcs, linestyle='--', color='k', linewidth=0.5)
    ax.text(num_retained_pcs+5, 0.2, f'# PC={num_retained_pcs}')
    ax.set_xlim([0,n_pc_displayed])
    sns.despine()

    return fig, ax

def grid_search_spectral_clustering(        
        retained_pc_data: np.ndarray,
        param_grid: Dict[str, Any],
        random_state: int,
        results_save_path: str
    ) -> pd.DataFrame:
    """
    Perform a grid search over different parameter combinations for Spectral Clustering
    and evaluate using silhouette scores.

    Parameters
    ----------
    retained_pc_data : np.ndarray
        The retained principal components data to be used for clustering.
    param_grid : dict
        Dictionary with parameters names (`n_clusters`, `n_nearest_neighbors`) as keys
        and lists of parameter settings to try as values.
    random_state : int
        Seed for the random number generator.
    results_save_path : str
        Path to save the results CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the clustering results and silhouette scores.
    """
    results: List[dict] = []
    clusters_options = param_grid.get('n_clusters', [])
    neighbors_options = param_grid.get('n_nearest_neighbors', [])
    total_iter = len(clusters_options) * len(neighbors_options)

    with tqdm(total=total_iter) as pbar:
        for n_clusters in clusters_options:
            for n_nn in neighbors_options:
                model = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='nearest_neighbors',
                    n_neighbors=n_nn,
                    random_state=random_state,
                    n_jobs=-1
                )
                model.fit(retained_pc_data)
                score = silhouette_score(
                    retained_pc_data,
                    model.labels_,
                    metric='cosine'
                )
                results.append({
                    'n_clusters': n_clusters,
                    'n_nearest_neighbors': n_nn,
                    'silhouette_score': score,
                    'cluster_labels': model.labels_
                })
                pbar.update(1)
                logging.info(f'Score = {score:.3f} for # clusters = {n_clusters}, # nearest neighbors = {n_nn}')

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_save_path, index=False)
    logging.info(f'Results saved to {results_save_path}')

    return results_df

def reorder_cluster_labels(
        labels_from_model: np.ndarray,
        population_activity: np.ndarray,
        n_pre_stim_frames: int,
        post_stim_window: int,
        save_path : str
        ) -> np.ndarray:
    """
    Since the clustering labels are arbitrary, we reorder cluster labels based on
    the sorted average response in a post-stimulus window.

    Parameters
    ----------
    labels_from_model : np.ndarray
        Array of cluster labels from the model.
    population_activity : np.ndarray
        Array of neuronal activity data where rows represent neurons and columns represent time points.
    n_pre_stim_frames : int
        Number of frames in the pre-stimulus period.
    post_stim_window : int
        Number of frames in the post-stimulus period to order the clusters.
    save_path : str
        Path to save the reordered_labels as `.npy` file.

    Returns
    -------
    np.ndarray
        Array of reordered cluster labels.
    """
    unique_labels = np.unique(labels_from_model)
    responses = np.full(len(unique_labels), np.nan)

    for label in unique_labels:
        mask = (labels_from_model == label)
        responses[label] = np.mean(population_activity[mask, n_pre_stim_frames : post_stim_window])

    # Sorting labels by the responses
    sorted_labels = np.argsort(responses)[::-1]
    label_mapping = {original: new for new, original in enumerate(sorted_labels)}

    # Reassigning clusters to new sorted labels
    reordered_labels = np.array([label_mapping[label] for label in labels_from_model])

    np.save(save_path, reordered_labels)
    logging.info(f'Results saved to {save_path}')

    return reordered_labels

def get_label_lesion_identity(
        cluster_labels: np.ndarray,
        n_non_lesioned: int) -> Dict[str, List[int]]:
    """
    Separate cluster labels into 'Non-lesioned' and 'Lesioned' based on the number of non-lesioned samples,
    assuming the first `n_non_lesioned` labels are for neurons from 'Non-lesioned' followed by those from 'lesioned'.

    Parameters
    ----------
    cluster_labels: np.ndarray
        A list of cluster labels.
    n_non_lesioned: int
        The number of non-lesioned samples.

    Returns
    ----------
    Dict[str, List[int]]
        A dictionary with keys 'Non-lesioned' and 'Lesioned' mapping to the respective cluster labels.
    """
    return {
        'Non-lesioned': cluster_labels[:n_non_lesioned],
        'Lesioned': cluster_labels[n_non_lesioned:]
    }

def create_autopct(values: List[float]) -> Callable[[float], str]:
    """
    Returns a function to format pie chart labels with percentage and actual count.

    Parameters
    ----------
    values: List[float]
        A list of numerical values representing pie chart slices.

    Returns
    ----------
    Callable[[float], str]
        A function that formats labels as 'percentage%\n(count)'.
    """
    def format_autopct(percentage: float) -> str:
        total = sum(values)
        count = int(round(percentage * total / 100.0))
        return f'{percentage:.0f}%\n({count:d})'

    return format_autopct

def plot_cluster_pie(
        cluster_labels: np.ndarray,
        label_lesion_identity: Dict[str, np.ndarray],
        colors: Dict[str, str]
    ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot pie charts for each cluster, showing the proportion and count of neurons from 'Non-lesioned' and 'Lesioned'.

    Parameters
    ----------
    cluster_labels: np.ndarray
        An array of cluster labels.
    label_lesion_identity: Dict[str, np.ndarray]
        A dictionary with keys 'Non-lesioned' and 'Lesioned', each containing an array of cluster labels.
    colors: Dict[str, str]
        A dictionary mapping 'Non-lesioned' and 'Lesioned' to their respective color codes.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        A tuple containing the figure and axes objects created by the plot.
    """
    unique_labels = sorted(np.unique(cluster_labels))
    plt.rc('font', size=14)
    fig, axes = plt.subplots(1, len(unique_labels), figsize=(2.5 * len(unique_labels), 5))

    for cluster, ax in zip(unique_labels, axes):
        n_neurons = {
            'Non-lesioned': sum(label_lesion_identity['Non-lesioned'] == cluster),
            'Lesioned': sum(label_lesion_identity['Lesioned'] == cluster)
        }
        ax.pie(
            x=list(n_neurons.values()),
            labels=None,
            colors=list(colors.values()),
            autopct=create_autopct(list(n_neurons.values()))
        )

    return fig, axes

def plot_clustered_heatmaps(
    population_activity: np.ndarray,
    cluster_labels: np.ndarray,
    n_imaging_frames_per_trial: int,
    trial_sec: int,
    n_pre_stim_frames: int,
    trial_types: List[str],
    sort_window: List[int],
    cmax: float = 1.0
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot heatmaps of neuronal activity for each cluster.

    Parameters
    ----------
    population_activity : np.ndarray
        2D array of neuronal activity data where rows represent neurons and columns represent time points.
    cluster_labels : np.ndarray
        Array of cluster labels corresponding to each neuron in `population_activity`.
    n_imaging_frames_per_trial : int
        Number of imaging frames in each trial.
    trial_sec : int
        Duration of the trial in seconds.
    n_pre_stim_frames : int
        Number of frames in the pre-stimulus period.
    trial_types : List[str]
        List of trial types to be plotted.
    sort_window : List[int]
        Start and end time points for sorting the neurons based on activity.
    cmax : float, optional
        Maximum absolute value for heatmap color scaling. Defaults to 1.0.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure object containing the plot.
    axes : np.ndarray
        Array of axes objects corresponding to the subplots.
    """
    unique_labels = sorted(np.unique(cluster_labels))
    n_clusters = len(unique_labels)
    n_trial_types = len(trial_types)

    # Set up the figure and axes
    fig, axes = plt.subplots(n_trial_types, n_clusters,
                            figsize=(3.2 * n_clusters, 6 * n_trial_types))
    cbar_ax = fig.add_axes([.94, .3, .01, .4])
    cbar_ax.tick_params(width=0.5)

    for cluster_index, cluster in enumerate(unique_labels):
        for trial_type_index in range(n_trial_types):
            neuron_indices = np.where(cluster_labels == cluster)[0]
            trial_start_index = trial_type_index * n_imaging_frames_per_trial
            trial_end_index = (trial_type_index + 1) * n_imaging_frames_per_trial
            cluster_activity = population_activity[neuron_indices, trial_start_index:trial_end_index] # recall that the population activity includes concatenated responses for hit and miss trials

            # Sort neurons for display based on mean activity in the specified window
            sorted_response = np.argsort(np.mean(cluster_activity[:, sort_window[0]:sort_window[1]], axis=1))[::-1]

            sns.heatmap(
                cluster_activity[sorted_response],
                ax=axes[trial_type_index, cluster_index],
                cmap=plt.get_cmap('BrBG_r'),
                vmin=-cmax,
                vmax=cmax,
                cbar=(cluster_index == 0),
                cbar_ax=cbar_ax if cluster_index == 0 else None,
                cbar_kws={'label': 'Normalized fluorescence'}
                )

            axes[trial_type_index, cluster_index].grid(False)
            axes[trial_type_index, cluster_index].tick_params(width=0.5)
            axes[trial_type_index, cluster_index].set_yticks([])

            if trial_type_index == n_trial_types - 1:
                x_ticks_loc = (np.linspace(-2, 5, trial_sec + 1) * n_imaging_frames_per_trial / trial_sec + n_imaging_frames_per_trial / trial_sec * 2)[::2]
                axes[trial_type_index, cluster_index].set_xticks(x_ticks_loc)
                axes[trial_type_index, cluster_index].set_xticklabels(np.linspace(-2, 5, trial_sec + 1).astype(int)[::2], rotation=0, fontsize=20)
            else:
                axes[trial_type_index, cluster_index].set_xticks([])

            axes[trial_type_index, cluster_index].axvline(n_pre_stim_frames, linestyle='--', color='k', linewidth=2, alpha=0.7) # dashed line showing stimulus onset

            if cluster_index == 0:
                axes[trial_type_index, 0].set_ylabel('Neurons', fontsize=20)

        axes[0, cluster_index].set_title(f'Cluster {cluster + 1}\n(n={cluster_activity.shape[0]})\n', fontsize=28)

    fig.text(0.5, 0.1, 'Time (s)', fontsize=28,
             horizontalalignment='center', verticalalignment='center', rotation='horizontal')
    fig.text(-0.015, 0.67, trial_types[0], fontsize=28, rotation='horizontal')
    fig.text(-0.02, 0.34, trial_types[1], fontsize=28, rotation='horizontal')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.subplots_adjust(left=0.03, right=0.93, bottom=0.2, top=0.83)

    return fig, axes

def plot_average_responses(
    population_activity: np.ndarray,
    cluster_labels: np.ndarray,
    n_imaging_frames_per_trial: int,
    trial_types: List[str],
    colors: Dict[str, str],
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot the average neuronal responses for each cluster and trial type.

    Parameters
    ----------
    population_activity : np.ndarray
        2D array of neuronal activity data where rows represent neurons and columns represent time points.
    cluster_labels : np.ndarray
        Array of cluster labels corresponding to each neuron in `population_activity`.
    n_imaging_frames_per_trial : int
        Number of imaging frames in each trial.
    trial_types : List[str]
        List of trial types to be plotted.
    colors : Dict[str, str]
        Dictionary mapping trial types to their respective plot colors.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure object containing the plot.
    axes : np.ndarray
        Array of axes objects corresponding to the subplots.
    """
    unique_labels = sorted(np.unique(cluster_labels))
    n_clusters = len(unique_labels)
    x_axis = np.linspace(-2, 5, n_imaging_frames_per_trial)
    fig, axes = plt.subplots(1, n_clusters, figsize=(3.2 * n_clusters, 3), sharey=True)

    for cluster_index, cluster in enumerate(unique_labels):
        ax = axes[cluster_index]
        ax.axvline(0, linestyle='--', color='k', linewidth=1, alpha=0.3) # dashed line showing stimulus onset
        for trial_type_index, trial_type in enumerate(trial_types):
            neuron_indices = np.where(cluster_labels == cluster)[0]
            trial_start_index = trial_type_index * n_imaging_frames_per_trial
            trial_end_index = (trial_type_index + 1) * n_imaging_frames_per_trial
            cluster_activity = population_activity[neuron_indices, trial_start_index:trial_end_index] # recall that the population activity includes concatenated responses for hit and miss trials
            ax.plot(
                x_axis,
                np.mean(cluster_activity, axis=0),
                linewidth=3,
                color=colors[trial_type]
                )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    sns.despine(left=True, bottom=True)

    return fig, axes