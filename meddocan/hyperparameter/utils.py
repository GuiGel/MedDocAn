from typing import Any, Dict, List

from click import BadParameter
from flair.embeddings import StackedEmbeddings

from meddocan.hyperparameter.parameter import Parameter


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
