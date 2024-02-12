from typing import Optional
import re

from .formatter import TermUnFormatter
from .sequencer import TemplateSequencer, TemplateSequencerResult
from ..base import Label, QAPrompt


class Surfacer:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(
        self,
        qa_prompt: QAPrompt,
        chosen_answer_label: Label,
        *args,
        **kwargs,
    ) -> str:
        raise NotImplementedError


class TextSurfacer(Surfacer):
    def __init__(self, prefix: str, text: str):
        super().__init__(prefix)
        self.text = text

    def __call__(self, *args, **kwargs) -> str:
        return self.prefix + self.text


class TemplateSurfacer(Surfacer):
    def __init__(
        self,
        prefix: str,
        un_formatter: Optional[TermUnFormatter] = None,
    ):
        super().__init__(prefix)
        self.un_formatter = un_formatter
        self.pos_neg_pattern = re.compile(r"(\[\[(.+?)\|(.+?)]])")

    def __call__(
        self,
        qa_prompt: QAPrompt,
        chosen_answer_label: Label,
        *args,
        **kwargs,
    ) -> str:
        # Grab the sequencer result.
        if "result" not in kwargs:
            raise KeyError("No PromptGroupSequencerResult provided for surfacing.")
        result: TemplateSequencerResult = kwargs["result"]

        # Optionally unformat the source and target terms.
        source_term = result.template.source.term
        if self.un_formatter is not None:
            source_term = self.un_formatter.unformat(source_term, *args, **kwargs)
        target_term = result.template.target.term
        if self.un_formatter is not None:
            target_term = self.un_formatter.unformat(target_term, *args, **kwargs)

        # Format the surface form using the source and target terms.
        text = result.template.relation.surface_form.format(source_term, target_term)

        # Find all positive/negative variations in the surface form of the template.
        matches = self.pos_neg_pattern.findall(text)

        # If the template is the pairing template of the corresponding tree, it MUST
        # contain variations. Replace them based on label match with the chosen answer.
        if result.template == qa_prompt.tree_map[result.tree_label].pairing_template:
            if not matches:
                raise ValueError(
                    "Pairing template must contain positive/negative variations."
                )
            is_positive = chosen_answer_label == result.tree_label
            for group, positive, negative in matches:
                text = text.replace(group, positive if is_positive else negative, 1)
        elif matches:
            # The template is not a pairing template, so it CANNOT contain variations.
            raise ValueError(
                "Non-pairing template cannot contain positive/negative variations."
            )
        return self.prefix + text


class TemplateSequenceSurfacer(Surfacer):
    def __init__(
        self,
        prefix: str,
        template_separator: str,
        surfacer: TemplateSurfacer,
        sequencer: TemplateSequencer,
    ):
        super().__init__(prefix)
        self.sequencer = sequencer
        self.surfacer = surfacer
        self.template_separator = template_separator

    def __call__(
        self,
        qa_prompt: QAPrompt,
        chosen_answer_label: Label,
        *args,
        **kwargs,
    ) -> str:
        def fn_caller(fn, **result):
            return fn(qa_prompt, chosen_answer_label, *args, **kwargs, **result)

        return self.prefix + self.template_separator.join(
            [fn_caller(self.surfacer, result=r) for r in fn_caller(self.sequencer)]
        )


class QADataSurfacer(Surfacer):
    def __init__(
        self,
        prefix: str,
        question_answer_separator: str,
        answer_choice_separator: str,
        answer_choice_formatter: str,
    ):
        super().__init__(prefix)
        self.question_answer_separator = question_answer_separator
        self.answer_choice_separator = answer_choice_separator
        self.answer_choice_formatter = answer_choice_formatter

    def __call__(self, qa_prompt: QAPrompt, *args, **kwargs) -> str:
        answer_choices = [
            self.answer_choice_formatter.format(label, term)
            for label, term in qa_prompt.qa_data.answer_choices.items()
        ]
        return (
            self.prefix
            + qa_prompt.qa_data.question
            + self.question_answer_separator
            + self.answer_choice_separator.join(answer_choices)
        )


class QAPromptSurfacer(Surfacer):
    def __init__(
        self,
        prefix: str,
        surfacer_separator: str,
        prefix_surfacer: Surfacer,
        template_sequence_surfacer: Surfacer,
        qa_data_surfacer: Surfacer,
        suffix_surfacer: Surfacer,
    ):
        super().__init__(prefix)
        self.prefix_surfacer = prefix_surfacer
        self.template_sequence_surfacer = template_sequence_surfacer
        self.qa_data_surfacer = qa_data_surfacer
        self.suffix_surfacer = suffix_surfacer
        self.surfacer_separator = surfacer_separator

    def __call__(
        self,
        qa_prompt: QAPrompt,
        chosen_answer_label: Label,
        *args,
        **kwargs,
    ) -> str:
        def fn_caller(fn):
            return fn(qa_prompt, chosen_answer_label, *args, **kwargs)

        return self.prefix + self.surfacer_separator.join(
            [
                fn_caller(self.prefix_surfacer),
                fn_caller(self.template_sequence_surfacer),
                fn_caller(self.qa_data_surfacer),
                fn_caller(self.suffix_surfacer),
            ]
        )
