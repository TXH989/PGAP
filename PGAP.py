# -*- coding: utf-8 -*-


from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PaperProtocol:
    model_name: str = "PGAP-DeepOKAN"
    operator_basis: str = "Chebyshev"
    polynomial_degree: int = 9
    branch_layers: int = 4
    trunk_layers: int = 4
    hidden_width: int = 20
    latent_dim_per_field: int = 20
    indicator_hidden_width: int = 32
    indicator_quantile: float = 0.90
    lambda_indicator: float = 0.1
    lambda_wall: float = 1.0
    learning_rate: float = 1e-3
    optimizer: str = "AdamW"
    scheduler: str = "cosine_annealing"
    weight_decay_main: float = 1e-6
    weight_decay_feedback: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 500
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    high_gradient_fraction: float = 0.10
    smooth_fraction: float = 0.50

    seeds: Tuple[int, ...] = ()
    split_seed: Optional[int] = None
    batch_size: Optional[int] = None
    max_epochs: Optional[int] = None


def interface(operation, *args, **kwargs):

    raise NotImplementedError(
        f"{operation}: core/data implementation omitted in this pseudocode."
    )


def train_one_run(protocol, seed, train_data, validation_data, normalization):
    interface("set_random_seed", seed)
    model = interface("build_PGAP_DeepOKAN", protocol)
    optimizer = interface("build_optimizer", model, protocol)
    scheduler = interface("build_scheduler", optimizer, protocol)
    best_validation = float("inf")
    best_weights = None
    no_improvement = 0

    for epoch in range(protocol.max_epochs):
        interface("enter_training_mode", model)
        for batch in interface("training_batches", train_data, protocol.batch_size):

            prediction, auxiliary = interface("model_forward", model, batch)
            loss = interface(
                "joint_training_objective", prediction, auxiliary,
                batch, normalization, protocol,
            )
            interface("zero_gradients", optimizer)
            interface("backpropagate", loss)
            interface("clip_gradient_norm", model, protocol.grad_clip)
            interface("optimizer_step", optimizer)

        interface("scheduler_step", scheduler)

        validation_loss = interface(
            "validation_field_loss", model, validation_data, normalization,
        )
        interface("record_epoch_metrics", epoch, validation_loss)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_weights = deepcopy(interface("model_state", model))
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= protocol.early_stop_patience:
            break

    interface("restore_model_state", model, best_weights)
    interface("save_checkpoint_with_config_and_normalization", model, protocol,
              seed, normalization)
    return model


def main(protocol):
    if len(protocol.seeds) != 5 or len(set(protocol.seeds)) != 5:
        raise ValueError("Provide the five distinct seeds from the actual experiments.")
    if protocol.split_seed is None:
        raise ValueError("Provide the recorded data-split seed.")
    if protocol.batch_size is None or protocol.batch_size <= 0:
        raise ValueError("Provide the actual positive training batch size.")
    if protocol.max_epochs is None or protocol.max_epochs <= 0:
        raise ValueError("Provide the actual maximum training epochs.")

    data = interface("load_and_validate_topology_aligned_RAE2822_data")
    train, validation, test = interface(
        "fixed_case_level_split", data, protocol.split_ratio, protocol.split_seed,
    )

    normalization = interface("fit_training_normalization", train)
    results = []
    for seed in protocol.seeds:
        model = train_one_run(protocol, seed, train, validation, normalization)

        metrics = interface("evaluate_held_out_test_set", model, test,
                            normalization, protocol)
        interface("save_run_metrics", seed, metrics)
        results.append(metrics)
    interface("summarize_five_runs_mean_and_standard_deviation", results)


if __name__ == "__main__":
    main(PaperProtocol())
