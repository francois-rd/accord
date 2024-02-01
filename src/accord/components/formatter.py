from ..base import Term


class TermFormatter:
    def format(self, term: Term, language: str, *args, **kwargs) -> Term:
        raise NotImplementedError
