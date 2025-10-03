from src.utils.activation_visualizer import ActivationVisualizer
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt


def get_data(languages):
    from datasets import load_dataset
    data = {}
    if isinstance(languages, DictConfig):
        for sub in languages:
            for language in languages[sub]:
                data[language] = load_dataset("Davlan/sib200", language, split="test", cache_dir='/mnt/data/wilsontansil/.cache/huggingface/datasets')
    else:
        for language in languages:
            data[language] = load_dataset("Davlan/sib200", language, split="test", cache_dir='/mnt/data/wilsontansil/.cache/huggingface/datasets')
    return data

def get_color_map(languages):
    cmap = plt.get_cmap(f"tab20")
    family_to_color = {
        family: cmap(i) for i, family in enumerate(languages)
    }
    return family_to_color

@hydra.main(config_path="../../config", config_name="plotting", version_base=None)
def main(
    cfg: DictConfig
) -> None:
    # assert models, "Please provide at least one model in the models argument"
    # assert languages, "Please provide at least one language in the languages argument"
    # assert color_map, "Please provide a color map in the color_map argument"
    
    # print(cfg.languages)
    cmap = get_color_map(cfg.languages)
    data = get_data(cfg.languages)
    activation = ActivationVisualizer(
        models=[cfg.models],
        languages=cfg.languages,
        data=data,
        color_map=cmap
    )

    activation.generate_plots_classification(
        save_path=cfg.save_path,
        ext=cfg.ext,
        activation_path="/mnt/data/lingua_franca/activation-extraction/outputs/topic_classification/",
        input_mode=cfg.input_mode,
        extraction_mode=cfg.extraction_mode,
        plot_by=cfg.plot_by,
        device=cfg.device
    )

if __name__ == "__main__":
    main()