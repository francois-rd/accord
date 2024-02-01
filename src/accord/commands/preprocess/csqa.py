from dataclasses import dataclass
from collections import Counter
import random
import os

from tqdm import tqdm
import pandas as pd

from .conceptnet import ConceptNetConfig
from ..configs import ResourcesConfig
from ...base import QAData, Relation, Template, Variable
from ...databases.conceptnet import ConceptNet
from ...io import (
    ensure_path,
    save_json,
    save_dataclass_jsonl,
    load_json,
    load_jsonl,
    load_records_csv,
)


@dataclass
class CSQAConfig:
    raw_data_file: str = "raw/dev_rand_split.jsonl"
    inferred_data_file: str = "inferred/dev.json"
    inference_failures_file: str = "inferred/failures.json"
    sub_sampled_data_file: str = "sampled/dev_${limit}_${seed}.csv"
    paired_data_dir: str = "paired/dev_${limit}_${seed}"
    converted_data_file: str = "converted/dev_${limit}_${seed}.jsonl"
    language: str = "en"
    limit: int = 20
    seed: int = 314159
    verbose: bool = False


def infer(resources: ResourcesConfig, cfg: CSQAConfig, c_net_cfg: ConceptNetConfig):
    """
    Infers the ConceptNet relation type of each sample in CSQA based on majority vote.
    """
    # Merge resource paths.
    raw_data_file = os.path.join(resources.qa_dataset_dir, cfg.raw_data_file)
    inferred_data_file = os.path.join(resources.qa_dataset_dir, cfg.inferred_data_file)
    failures_file = os.path.join(resources.qa_dataset_dir, cfg.inference_failures_file)
    conceptnet_d = os.path.join(resources.term_database_dir, c_net_cfg.preprocessed_dir)

    # Ensure language consistency.
    if cfg.language not in c_net_cfg.languages:
        raise ValueError("CSQA is English-only, but ConceptNet preprocess removed it.")

    # Load ConceptNet.
    concept_net = ConceptNet(conceptnet_d, c_net_cfg.relation_map)

    # Load the raw CSQA data.
    csqa_data = load_jsonl(raw_data_file)

    # Infer the ConceptNet relation of each sample in CSQA based on majority vote.
    csqa_by_type = {}
    failures = []
    for datum in tqdm(csqa_data, desc="Progress", disable=not cfg.verbose):
        question = datum['question']
        node_word = concept_net.format(question['question_concept'], cfg.language)
        other_words = [choice['text'] for choice in question['choices']]
        relations = []
        for other_word in other_words:
            other_word = concept_net.format(other_word, cfg.language)
            relations.extend(concept_net.get_relations(node_word, other_word))
        histogram = Counter(relations).most_common()
        if histogram:
            csqa_by_type.setdefault(histogram[0][0], []).append(datum)
        else:
            failures.append(datum)

    # Save the CSQA-by-relation-type and the failure data.
    save_json(inferred_data_file, csqa_by_type)
    save_json(failures_file, failures, indent=4)


def sample(resources: ResourcesConfig, cfg: CSQAConfig):
    """
    Given CSQA data split by inferred ConceptNet relation type, for each such relation
    type, sub-samples up to 'limit' items from the associated CSQA samples and saves
    the sub-sampled data to file in CSV format.
    """
    # Merge resource paths.
    inferred_file = os.path.join(resources.qa_dataset_dir, cfg.inferred_data_file)
    sampled_file = os.path.join(resources.qa_dataset_dir, cfg.sub_sampled_data_file)

    # Load the CSQA-by-relation-type data.
    csqa_by_type = load_json(inferred_file)

    # For each relation type (key), sub-sample up to 'limit' items from the
    # corresponding list of CSQA samples (values).
    sub_samples = {}
    random.seed(cfg.seed)
    for k, values_list in csqa_by_type.items():
        if len(values_list) <= cfg.limit:
            sub_samples[k] = values_list
        else:
            sub_samples[k] = random.sample(values_list, cfg.limit)
        if cfg.verbose:
            print(f"{k}: sub-sampling {len(sub_samples[k])} of {len(values_list)}")

    # Save the data to CSV in a useful output format.
    df_data = {}
    for relation_type, relation_data in sub_samples.items():
        for datum in relation_data:
            df_data.setdefault('id', []).append(datum['id'])
            df_data.setdefault('type', []).append(relation_type)
            df_data.setdefault('concept', []).append(
                datum['question']['question_concept']
            )
            df_data.setdefault('answer', []).append(datum['answerKey'])
            df_data.setdefault('stem', []).append(datum['question']['stem'])
            for choice in datum['question']['choices']:
                df_data.setdefault(choice['label'], []).append(choice['text'])
    pd.DataFrame(df_data).to_csv(ensure_path(sampled_file), sep='\t', index=False)


def convert(resources: ResourcesConfig, cfg: CSQAConfig):
    """
    Converts the sub-sampled CSQA data from CSV to JSONL format as a generic interface.
    The input CSV format should match the output format of 'sub_sample_csqa()', but with
    the addition of pairing template data. The expectation is that some rows will have
    been manually culled from the sub-sampled file, as well.
    """
    # Merge resource paths.
    sampled_file = os.path.join(resources.qa_dataset_dir, cfg.sub_sampled_data_file)
    paired_data_dir = os.path.join(resources.qa_dataset_dir, cfg.paired_data_dir)
    converted_file = os.path.join(resources.qa_dataset_dir, cfg.converted_data_file)

    # Convert the data from CSV text to JSONL dataclasses.
    qa_dataset = []
    for record in load_records_csv(sampled_file, sep='\t'):
        # Load the appropriate pairing templates or reject QAData with no pairings.
        templates_file = os.path.join(paired_data_dir, f"{record['id']}.csv")
        if not os.path.isfile(templates_file):
            continue

        # Convert each template's data into a ReasoningTemplate.
        templates = []
        for template in load_records_csv(templates_file):
            placement, term = template.pop('placement'), template.pop('term')
            if placement == 'SOURCE':
                source = Variable(identifier="pairing_source", term=term)
                target = Variable(identifier="pairing_target")
            elif placement == 'TARGET':
                source = Variable(identifier="pairing_source")
                target = Variable(identifier="pairing_target", term=term)
            else:
                raise ValueError(f"Unsupported value for 'placement': {placement}")
            templates.append(Template(source, Relation(**template), target))

        # Convert all record data into a QAData object.
        qa_data = QAData(
            identifier=record.pop('id'),
            question=record.pop('stem'),
            correct_answer_label=record.pop('answer'),
            pairing_templates=templates,
            answer_choices=record,
            kwargs={
                "concept_net_relation_type": record.pop('type'),
                "csqa_question_concept": record.pop('concept'),
            },
        )
        qa_dataset.append(qa_data)
    save_dataclass_jsonl(converted_file, *qa_dataset)
