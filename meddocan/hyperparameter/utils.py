from typing import Any, Dict, List

from click import BadParameter
from flair.embeddings import StackedEmbeddings
from flair.nn.model import Model

from meddocan.hyperparameter.parameter import Parameter, ParameterName


def get_tensorboard_dirname(params: Dict[str, Any]) -> str:
    """Return a compact representation of a parameter dict.

    Args:
        params (Dict[str, Any]): Dict of parameters.

    Returns:
        str: The name of the folder for tensorboard usage.
    """
    names: List[str] = []
    admissible_params = [p.value for p in Parameter]

    for k, v in params.items():

        if k not in admissible_params:
            raise BadParameter(
                f"The params key {k} is not a valid parameter.\n"
                f"Key must be in one of {admissible_params}."
            )

        k = getattr(ParameterName, k).value  # reduce k length

        # ---- Included only the ParameterName that are not None
        if k is not None:

            if hasattr(v, "__name__"):
                name = f"{k}_{v.__name__}"
            elif hasattr(v, "name"):

                def get_emb_name(emb):
                    return emb.name.split("/")[-1]

                if isinstance(v, StackedEmbeddings):
                    e = ", ".join(
                        [
                            f"{i}_{get_emb_name(emb)}"
                            for i, emb in enumerate(v.embeddings)
                        ]
                    )
                    name = f"{k}_{v.name}({e})"
                else:
                    name = f"{k}_{get_emb_name(v)}"
            else:
                name = f"{k}_{v}"

            names.append(name)

    return "_".join(names)


def get_model_card(model: Model) -> str:
    if hasattr(model, "model_card"):
        param_out = "\n------------------------------------\n"
        param_out += "--------- Flair Model Card ---------\n"
        param_out += "------------------------------------\n"
        param_out += "- this Flair model was trained with:  \n"
        param_out += f"- Flair version {model.model_card['flair_version']}\n"
        param_out += (
            f"- PyTorch version {model.model_card['pytorch_version']}\n"
        )
        if "transformers_version" in model.model_card:
            param_out += (
                "- Transformers version "
                f"{model.model_card['transformers_version']}\n"
            )
        param_out += "------------------------------------\n"

        param_out += "------- Training Parameters: -------\n"
        param_out += "------------------------------------\n"
        training_params = "\n".join(
            f'- {param} = {model.model_card["training_parameters"][param]}'
            for param in model.model_card["training_parameters"]
        )
        param_out += training_params + "\n"
        param_out += "------------------------------------\n"

    else:
        param_out = (
            "This model has no model card (likely because it is not yet "
            "trained or was trained with Flair version < 0.9.1)"
        )
    return param_out
