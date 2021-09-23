import umap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


colormap = (
    np.array(
        [
            [76, 255, 0],
            [0, 127, 70],
            [255, 0, 0],
            [255, 217, 38],
            [0, 135, 255],
            [165, 0, 165],
            [255, 167, 255],
            [0, 255, 255],
            [255, 96, 38],
            [142, 76, 0],
            [33, 0, 127],
            [0, 0, 0],
            [183, 183, 183],
        ],
        dtype=np.float,
    )
    / 255
)

# assumption for embeddings: points for each speaker of length num_utter_per_speaker lined up, one speaker after another
def plot_embeddings(embeddings, num_utter_per_speaker, labels, max_speaker=10, max_utter=10): 
    embeddings = embeddings.reshape(embeddings.shape[0] // num_utter_per_speaker, num_utter_per_speaker, -1)
    num_utter_per_speaker = min(num_utter_per_speaker, max_utter)
    embeddings = embeddings[: max_speaker, : max_utter].reshape(-1, embeddings.shape[-1])
    # embeddings = embeddings[: max_speaker * num_utter_per_speaker] # [100, 256]
    model = umap.UMAP()
    projection = model.fit_transform(embeddings) #[x, 2]
    num_speakers = embeddings.shape[0] // num_utter_per_speaker
    ground_truth = np.repeat(np.arange(num_speakers), num_utter_per_speaker) # [100], [0,0,0,...,1,1,1,...]
    colors = [colormap[i] for i in ground_truth]

    fig, ax = plt.subplots(figsize=(16, 10))
    for i in range(num_speakers):
        points = projection[num_utter_per_speaker*i:num_utter_per_speaker*(i+1)]
        ax.scatter(points[:,0], points[:,1], color=colormap[i], label=np.array(labels)[:,0][i])
        
    ax.legend()

    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection")
    plt.tight_layout()
    plt.savefig("umap")
    return fig
