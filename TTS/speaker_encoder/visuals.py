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


def plot_embeddings(embeddings, num_utter_per_speaker, labels, max_speaker=10, max_utter=10): # [500, 256]
    embeddings = embeddings.reshape(embeddings.shape[0] // num_utter_per_speaker, num_utter_per_speaker, -1)
    num_utter_per_speaker = min(num_utter_per_speaker, max_utter)
    embeddings = embeddings[: max_speaker, : max_utter].reshape(-1, embeddings.shape[-1])
    # embeddings = embeddings[: max_speaker * num_utter_per_speaker] # [100, 256]
    model = umap.UMAP()
    projection = model.fit_transform(embeddings) #[x, 2]
    num_speakers = embeddings.shape[0] // num_utter_per_speaker
    print(f'will plot {num_speakers} speakers with {num_utter_per_speaker} utterances each')
    ground_truth = np.repeat(np.arange(num_speakers), num_utter_per_speaker) # [100], [0,0,0,...,1,1,1,...]
    colors = [colormap[i] for i in ground_truth]

    fig, ax = plt.subplots(figsize=(16, 10))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=colors)
    x=np.array(labels)[:,0].tolist()
    ax.legend(labels=np.array(labels)[:,0].tolist(), loc="upper right", title="Classes")  
    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection")
    plt.tight_layout()
    plt.savefig("umap")
    return fig
