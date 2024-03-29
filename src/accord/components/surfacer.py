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


class TermSurfacer(Surfacer):
    def __init__(
        self,
        prefix: str,
        suffix: str,
        un_formatter: Optional[TermUnFormatter] = None,
    ):
        super().__init__(prefix)
        self.suffix = suffix
        self.un_formatter = un_formatter

    def __call__(self, *args, **kwargs) -> str:
        # Grab the Term.
        if "term" not in kwargs:
            raise KeyError("No Term provided for surfacing.")
        term = kwargs["term"]

        # Optionally, unformat the term.
        if self.un_formatter is not None:
            term = self.un_formatter.unformat(*args, **kwargs)  # Kwargs has 'term'.

        # Surface the term.
        return self.prefix + term + self.suffix


class TemplateSurfacer(Surfacer):
    def __init__(
        self,
        prefix: str,
        term_surfacer: Surfacer,
    ):
        super().__init__(prefix)
        self.surfacer = term_surfacer
        self.pos_neg_pattern = re.compile(r"(\[\[(.+?)\|(.+?)]])")

    def __call__(
        self,
        qa_prompt: QAPrompt,
        chosen_answer_label: Label,
        *args,
        **kwargs,
    ) -> str:
        if qa_prompt.tree_map is None:
            raise ValueError("No templates provided for surfacing.")

        # Grab the sequencer result.
        if "result" not in kwargs:
            raise KeyError("No PromptGroupSequencerResult provided for surfacing.")
        result: TemplateSequencerResult = kwargs["result"]

        # Surface the source and target terms.
        source_term = self.surfacer(*args, term=result.template.source.term, **kwargs)
        target_term = self.surfacer(*args, term=result.template.target.term, **kwargs)

        # Format the surface form using the surfaced source and target terms.
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
        template_surfacer: Surfacer,
        sequencer: TemplateSequencer,
    ):
        super().__init__(prefix)
        self.sequencer = sequencer
        self.surfacer = template_surfacer
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
        prefix_surfacer: Optional[Surfacer],
        template_sequence_surfacer: Optional[Surfacer],
        qa_data_surfacer: Optional[Surfacer],
        suffix_surfacer: Optional[Surfacer],
    ):
        super().__init__(prefix)
        self.surfacer_separator = surfacer_separator
        self.surfacers = [
            prefix_surfacer,
            template_sequence_surfacer,
            qa_data_surfacer,
            suffix_surfacer,
        ]

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
            [fn_caller(f) for f in self.surfacers if f is not None],
        )
