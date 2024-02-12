from itertools import combinations
from dataclasses import dataclass
from typing import Callable, List
import random

from ..base import Label, QAPrompt, Template


DuplicateTemplateFunc = Callable[[Template, Template], bool]


def default_duplicate_template_fn(t1: Template, t2: Template) -> bool:
    return t1.partial_equals(
        t2,
        source_terms=True,
        relation_types=True,
        relation_surface_forms=True,
        target_terms=True,
    )


@dataclass
class TemplateSequencerResult:
    tree_label: Label
    template: Template


class TemplateSequencer:
    def __init__(
        self,
        chosen_answer_position: int = -1,
        shuffle_tree_order: bool = False,
        shuffle_within_tree_template_order: bool = False,
        shuffle_between_tree_template_order: bool = False,
        remove_duplicate_templates: bool = False,
        duplicate_template_fn: DuplicateTemplateFunc = default_duplicate_template_fn,
    ):
        self.chosen_answer_position = chosen_answer_position
        self.shuffle_tree_order = shuffle_tree_order
        self.shuffle_within_tree_template_order = shuffle_within_tree_template_order
        self.shuffle_between_tree_template_order = shuffle_between_tree_template_order
        self.remove_duplicate_templates = remove_duplicate_templates
        self.duplicate_template_fn = duplicate_template_fn

    def __call__(
        self,
        qa_prompt: QAPrompt,
        chosen_answer_label: Label,
        *args,
        **kwargs,
    ) -> List[TemplateSequencerResult]:
        # If a specific position is given for the answer, position it appropriately.
        # Otherwise, begin by positioning all trees in answer choice order, then
        # shuffling if needed.
        trees = []
        for label in qa_prompt.qa_data.answer_choices:
            if self.chosen_answer_position > -1 and label == chosen_answer_label:
                continue
            trees.append((label, qa_prompt.tree_map[label]))
        if self.shuffle_tree_order:
            random.shuffle(trees)
        if self.chosen_answer_position > -1:
            trees.insert(
                self.chosen_answer_position,
                (chosen_answer_label, qa_prompt.tree_map[chosen_answer_label]),
            )

        # Expand each tree into its list of templates. The default is to have the
        # templates in order. If 'shuffle_within_tree_template_order' is True, then
        # templates within each tree are shuffled (but overall the templates are still
        # in tree order). If 'shuffle_between_tree_template_order' is True, then all
        # templates across all trees are shuffled.
        results = []
        for label, tree in trees:
            tree_results = [
                TemplateSequencerResult(label, template)
                for template in tree.templates
            ]
            if self.shuffle_within_tree_template_order:
                random.shuffle(tree_results)
            results.extend(tree_results)
        if self.shuffle_between_tree_template_order:
            random.shuffle(results)

        # If desired, remove duplicate templates.
        if self.remove_duplicate_templates:
            to_remove = set()
            for r1, r2 in combinations(results, 2):
                if self.duplicate_template_fn(r1.template, r2.template):
                    to_remove.add(r2)
            results = [result for result in results if result not in to_remove]

        # Return the final sequence of templates.
        return results
