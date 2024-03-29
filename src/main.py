from collections import namedtuple
from typing import Any, Dict
import random
import os

import coma

from accord.base import QAData
from accord.commands import preprocess, generate, prompt, analyze, configs as cfgs
from accord.io import load_dataclass_jsonl
from accord.transforms import mapping_distance_factory
from accord.components import (
    default_duplicate_template_fn,
    SemanticDistanceSorter,
    RandomUnSorter,
)
from accord.databases.conceptnet import (
    ConceptNet,
    ConceptNetInstantiator,
    ConceptNetFormatter,
    ConceptNetUnFormatter,
    InstantiatorVariant,
)
from accord.llms import (
    DummyConfig,
    DummyLLM,
    OpenAIConfig,
    OpenAILLM,
    TransformersConfig,
    TransformersLLM,
)


ConfigData = namedtuple("ConfigData", "id_ type_")
basic_analysis = ConfigData("basic_analysis", analyze.basic.BasicAnalysisConfig)
beam_search = ConfigData("beam_search", cfgs.BeamSearchConfig)
conceptnet = ConfigData("conceptnet", preprocess.conceptnet.ConceptNetConfig)
csqa = ConfigData("csqa", preprocess.csqa.CSQAConfig)
filt = ConfigData("filter", cfgs.FilterConfig)
general = ConfigData("general", cfgs.GeneralConfig)
mapping_distance = ConfigData("mapping_distance", cfgs.MappingDistanceConfig)
reducer = ConfigData("reducer", cfgs.ReducerConfig)
resources = ConfigData("resources", cfgs.ResourcesConfig)
sorter = ConfigData("sorter", cfgs.SorterConfig)
surfacer = ConfigData("surfacer", cfgs.QAPromptSurfacerConfig)

dummy = ConfigData("dummy", DummyConfig)
openai = ConfigData("openai", OpenAIConfig)
transformers = ConfigData("transformers", TransformersConfig)


def as_dict(*cfgs_data: ConfigData):
    return {cfg.id_: cfg.type_ for cfg in cfgs_data}


@coma.hooks.hook
def forest_csqa_conceptnet_init_hook(configs: Dict[str, Any]) -> Any:
    # Grab the initialized configs.
    general_cfg: cfgs.GeneralConfig = configs[general.id_]
    srcs_cfg: cfgs.ResourcesConfig = configs[resources.id_]
    csqa_cfg: preprocess.csqa.CSQAConfig = configs[csqa.id_]
    c_net_cfg: preprocess.conceptnet.ConceptNetConfig = configs[conceptnet.id_]
    sorter_cfg = cfgs.SorterConfig = configs[sorter.id_]

    # Remove the superfluous configs from the initialization of the command.
    factory = coma.hooks.init_hook.positional_factory
    init_hook = factory(csqa.id_, conceptnet.id_, sorter.id_)

    # Load ConceptNet.
    conceptnet_d = os.path.join(srcs_cfg.term_database_dir, c_net_cfg.preprocessed_dir)
    c_net = ConceptNet(conceptnet_d, c_net_cfg.relation_map)

    # Load the ConceptNet Instantiators.
    factual = ConceptNetInstantiator(
        concept_net=c_net,
        language=csqa_cfg.language,
        variant=InstantiatorVariant.FACTUAL,
    )
    anti_factual = ConceptNetInstantiator(
        concept_net=c_net,
        language=csqa_cfg.language,
        variant=InstantiatorVariant.ANTI_FACTUAL,
        method=c_net_cfg.anti_factual_method,
    )

    # Load the InstantiatorResultsSorter.
    if sorter_cfg.sorter == "semantic_distance":
        if sorter_cfg.semantic_distance_calculator == "":
            pass
        else:
            raise ValueError(
                f"Unsupported value for semantic distance calculator: "
                f"{sorter_cfg.semantic_distance_calculator}"
            )
        if sorter_cfg.semantic_distance_aggregator in ["sum", "mean"]:
            aggregator = sum  # Sum and mean are the same in this case.
        elif sorter_cfg.semantic_distance_aggregator == ["min"]:
            aggregator = min
        else:
            raise ValueError(
                f"Unsupported value for semantic distance aggregator: "
                f"{sorter_cfg.semantic_distance_aggregator}"
            )
        results_sorter = SemanticDistanceSorter(
            target_distance=sorter_cfg.semantic_distance_target,
            semantic_distance_calculator=...,
            distance_aggregator=aggregator,
        )
    elif sorter_cfg.sorter == "random":
        random.seed(general_cfg.random_seed)
        results_sorter = RandomUnSorter()
    else:
        raise ValueError(f"Unsupported value for sorter: {sorter_cfg.sorter}")

    # Use the factory to create an appropriate command.
    converted_file = os.path.join(srcs_cfg.qa_dataset_dir, csqa_cfg.converted_data_file)
    command = generate.forest.factory(
        qa_dataset_loader=lambda: load_dataclass_jsonl(converted_file, QAData),
        factual_instantiator_loader=lambda: factual,
        anti_factual_instantiator_loader=lambda: anti_factual,
        formatter_loader=lambda: c_net,
        sorter_loader=lambda: results_sorter,
        language=csqa_cfg.language,
    )

    # Initialize the command.
    return init_hook(command=command, configs=configs)


@coma.hooks.hook
def group_csqa_conceptnet_init_hook(configs: Dict[str, Any]) -> Any:
    # Grab the initialized configs.
    general_cfg: cfgs.GeneralConfig = configs[general.id_]
    srcs_cfg: cfgs.ResourcesConfig = configs[resources.id_]
    dist_cfg: cfgs.MappingDistanceConfig = configs[mapping_distance.id_]
    csqa_cfg: preprocess.csqa.CSQAConfig = configs[csqa.id_]

    # Set the random seed as early as possible.
    random.seed(general_cfg.random_seed)

    # Remove the superfluous configs from the initialization of the command.
    init_hook = coma.hooks.init_hook.positional_factory(mapping_distance.id_, csqa.id_)

    # Use the factory to create an appropriate mapping distance function (if any).
    fn = None
    if not dist_cfg.ignore:
        fn = mapping_distance_factory(
            target_distances=dist_cfg.target_distances,
            count_answer_ids=dist_cfg.count_answer_ids,
            count_pairing_ids=dist_cfg.count_pairing_ids,
        )

    # Use the factory to create an appropriate command.
    converted_file = os.path.join(srcs_cfg.qa_dataset_dir, csqa_cfg.converted_data_file)
    command = generate.group.factory(
        qa_dataset_loader=lambda: load_dataclass_jsonl(converted_file, QAData),
        formatter_loader=lambda: ConceptNetFormatter(),
        language=csqa_cfg.language,
        mapping_distance_fn=fn,
    )

    # Initialize the command.
    return init_hook(command=command, configs=configs)


@coma.hooks.hook
def prompt_csqa_conceptnet_init_hook(name: str, configs: Dict[str, Any]) -> Any:
    # Grab the LLM info.
    if "dummy" in name:
        llm_cfg_id = dummy.id_
        llm_class = DummyLLM
    elif "openai" in name:
        llm_cfg_id = openai.id_
        llm_class = OpenAILLM
    elif "transformers" in name:
        llm_cfg_id = transformers.id_
        llm_class = TransformersLLM
    else:
        raise ValueError(f"Unsupported prompt command: {name}")

    # Grab the initialized configs.
    general_cfg: cfgs.GeneralConfig = configs[general.id_]
    srcs_cfg: cfgs.ResourcesConfig = configs[resources.id_]
    csqa_cfg: preprocess.csqa.CSQAConfig = configs[csqa.id_]
    llm_cfg = configs[llm_cfg_id]

    # Set the random seed as early as possible.
    random.seed(general_cfg.random_seed)

    # Remove the superfluous configs from the initialization of the command.
    init_hook = coma.hooks.init_hook.positional_factory(csqa.id_, llm_cfg_id)

    # Use the factory to create an appropriate duplicate template function (if any).
    fn = default_duplicate_template_fn

    # Use the factory to create an appropriate command.
    converted_file = os.path.join(srcs_cfg.qa_dataset_dir, csqa_cfg.converted_data_file)
    command = prompt.factory(
        qa_dataset_loader=lambda: load_dataclass_jsonl(converted_file, QAData),
        un_formatter_loader=lambda: ConceptNetUnFormatter(),
        llm_loader=lambda: llm_class(srcs_cfg.llm, llm_cfg),
        duplicate_template_fn=fn,
    )

    # Initialize the command.
    return init_hook(command=command, configs=configs)


@coma.hooks.hook
def analyze_basic_csqa_conceptnet_init_hook(configs: Dict[str, Any]) -> Any:
    # Grab the initialized configs.
    srcs_cfg: cfgs.ResourcesConfig = configs[resources.id_]
    csqa_cfg: preprocess.csqa.CSQAConfig = configs[csqa.id_]

    # Remove the superfluous configs from the initialization of the command.
    init_hook = coma.hooks.init_hook.positional_factory(csqa.id_)

    # Use the factory to create an appropriate command.
    converted_file = os.path.join(srcs_cfg.qa_dataset_dir, csqa_cfg.converted_data_file)
    command = analyze.basic.factory(
        qa_dataset_loader=lambda: load_dataclass_jsonl(converted_file, QAData),
    )

    # Initialize the command.
    return init_hook(command=command, configs=configs)


@coma.hooks.hook
def pre_run_hook(known_args):
    if known_args.dry_run:
        print("Dry run.")
        quit()


if __name__ == "__main__":
    # Initialize.
    dry_run_hook = coma.hooks.parser_hook.factory(
        "--dry-run",
        action="store_true",
        help="exit during pre-run",
    )
    coma.initiate(
        parser_hook=coma.hooks.sequence(coma.hooks.parser_hook.default, dry_run_hook),
        pre_run_hook=pre_run_hook,
        **as_dict(resources, general),
    )

    # Data preprocessing commands.
    coma.register(
        "preprocess.conceptnet",
        preprocess.conceptnet.preprocess,
        **as_dict(conceptnet),
    )
    coma.register(
        "preprocess.csqa.infer",
        preprocess.csqa.infer,
        **as_dict(csqa, conceptnet),
    )
    coma.register(
        "preprocess.csqa.sample",
        preprocess.csqa.sample,
        **as_dict(csqa),
    )
    coma.register(
        "preprocess.csqa.convert",
        preprocess.csqa.convert,
        **as_dict(csqa),
    )

    # Generation commands.
    coma.register("generate.generic", generate.generic.generate, **as_dict(filt))
    coma.register(
        "generate.relational",
        generate.relational.Generate,
        **as_dict(reducer, filt),
    )
    with coma.forget(init_hook=True):
        coma.register(
            "generate.forest.csqa.conceptnet",
            generate.forest.placeholder,
            init_hook=forest_csqa_conceptnet_init_hook,
            **as_dict(beam_search, reducer, filt, csqa, conceptnet, sorter),
        )
    with coma.forget(init_hook=True):
        coma.register(
            "generate.group.csqa.conceptnet",
            generate.group.placeholder,
            init_hook=group_csqa_conceptnet_init_hook,
            **as_dict(beam_search, mapping_distance, csqa),
        )

    # Prompt commands.
    with coma.forget(init_hook=True):
        coma.register(
            "prompt.dummy.csqa.conceptnet",
            prompt.placeholder,
            init_hook=prompt_csqa_conceptnet_init_hook,
            **as_dict(filt, surfacer, csqa, dummy),
        )
        coma.register(
            "prompt.openai.csqa.conceptnet",
            prompt.placeholder,
            init_hook=prompt_csqa_conceptnet_init_hook,
            **as_dict(filt, surfacer, csqa, openai),
        )
        coma.register(
            "prompt.transformers.csqa.conceptnet",
            prompt.placeholder,
            init_hook=prompt_csqa_conceptnet_init_hook,
            **as_dict(filt, surfacer, csqa, transformers),
        )

    # Analysis commands.
    with coma.forget(init_hook=True):
        coma.register(
            "analyze.basic.csqa.conceptnet",
            analyze.basic.placeholder,
            init_hook=analyze_basic_csqa_conceptnet_init_hook,
            **as_dict(basic_analysis, csqa),
        )

    # Run.
    coma.wake()
