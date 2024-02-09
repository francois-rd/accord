from ..base import Term


class TermFormatter:
    def format(self, term: Term, language: str, *args, **kwargs) -> Term:
        raise NotImplementedError


class TermUnFormatter:
    def unformat(self, term: Term, *args, **kwargs) -> Term:
        raise NotImplementedError
