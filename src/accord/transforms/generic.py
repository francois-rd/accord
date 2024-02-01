from typing import Dict, Iterable, Optional

import networkx as nx

from ..base import (
    Case,
    GenericCaseLink,
    GenericTemplate,
    GenericTree,
    GenericVariable,
    RelationId,
)


class GenericTreeTransform:
    def __init__(self):
        """
        For a given sequence of n GenericCaseLinks, creates a unique GenericTemplate for
        each generic relation in the GenericCaseLinks (detecting duplicates via generic
        relation identifiers) and keeps track of their GenericVariables.

        From there, a GenericTree can be built by linking GenericVariables according
        to the GenericCaseLink information and then resolving GenericVariable parents.
        Finally, the GenericTree's validity can be checked and the results cleaned up.
        """
        self.templates = self.case_links = None

    def __call__(self, case_links: Iterable[GenericCaseLink]) -> Optional[GenericTree]:
        """
        Builds and returns a GenericTree (or None if invalid).
        """
        self.case_links = case_links
        self.templates = self._create_templates()
        try:
            self._link_variables()
        except ValueError:
            return None
        self._resolve_parents()
        if self._is_valid():
            return self._cleanup()
        return None

    def _create_templates(self) -> Dict[RelationId, GenericTemplate]:
        """
        Creates GenericTemplates out of a sequence of GenericCaseLinks. GenericTemplates
        are uniquely identified from their associated generic relation's identifier.
        """
        templates = {}
        variable_counter = 0
        for case_link in self.case_links:
            if case_link.r1_id not in templates:
                templates[case_link.r1_id] = GenericTemplate(
                    GenericVariable(identifier=f"V{variable_counter}"),
                    case_link.r1_id,
                    GenericVariable(identifier=f"V{variable_counter + 1}"),
                )
                variable_counter += 2
            if case_link.r2_id not in templates:
                templates[case_link.r2_id] = GenericTemplate(
                    GenericVariable(identifier=f"V{variable_counter}"),
                    case_link.r2_id,
                    GenericVariable(identifier=f"V{variable_counter + 1}"),
                )
                variable_counter += 2
        return templates

    def _link_variables(self):
        """
        Link GenericVariables to each other according to GenericCaseLink information.
        Raises TypeError on unsupported Case or ValueError on detection of multiple
        parents.

        Multiple parents are not allowed since it creates a derivation conflict where
        a single GenericVariable is meant to take on the value of multiple others.
        """
        for case_link in self.case_links:
            self._do_link_variables(case_link)

    def _do_link_variables(self, case_link: GenericCaseLink):
        """
        Link a specific GenericVariable pair according to the given GenericCaseLink
        information. Raises TypeError on unsupported Case or ValueError on detection
        of multiple parents.
        """
        main_template = self.templates[case_link.r1_id]
        linked_template = self.templates[case_link.r2_id]
        if case_link.case == Case.ZERO:
            return
        elif case_link.case == Case.ONE:
            main_var, linked_var = main_template.target, linked_template.source
        elif case_link.case == Case.TWO:
            main_var, linked_var = main_template.target, linked_template.target
        elif case_link.case == Case.THREE:
            main_var, linked_var = main_template.source, linked_template.source
        elif case_link.case == Case.FOUR:
            main_var, linked_var = main_template.source, linked_template.target
        else:
            raise TypeError(f"Unsupported case: {case_link.case}")

        main_var.has_children = True
        if linked_var.parent is None:
            linked_var.parent = main_var
        else:
            raise ValueError("Invalid. Multiple parents detected.")

    def _resolve_parents(self):
        """Recursively replaces all variables' parents with their parent's parent."""
        for template in self.templates.values():
            self._resolve_parent(template.source)
            self._resolve_parent(template.target)

    def _resolve_parent(self, variable: GenericVariable) -> GenericVariable:
        """Recursively replaces a variable's parent with its parent's parent."""
        if variable.parent is None:
            return variable
        variable.parent = self._resolve_parent(variable.parent)
        return variable.parent

    def _is_valid(self) -> bool:
        """
        Returns whether this builder's current data is a logically valid GenericTree.
        Specifically, only poly-trees (connected, undirected, acyclic graphs) are valid
        GenericTrees. Technically, there are some valid GenericTrees that are cyclic.
        For example:
            A rel1 B
            |\
            | A rel2 C
            |/
            A rel3 D
        However, in all these cases, a semantically-equivalent acyclic GenericTree can
        be constructed by removing one of the GenericCaseLinks. For example:
            A rel1 B
             \
              A rel2 C
             /
            A rel3 D
        or
            A rel1 B
            |\
            | A rel2 C
            |
            A rel3 D
        are both semantically-equivalent to the above and also acyclic. Rather than
        trying to handle these special valid cyclic cases, we apply an overly-cautious
        validity check (namely, acyclicity), and assume that the user is able to
        remove the appropriate edges of an otherwise-valid cyclic graph to make it
        acyclic while maintaining semantic-equivalency (or alternatively, that all
        graphs are exhaustively constructed such that a semantically-equivalent graph
        will pass the test eventually).
        """
        graph = nx.Graph()
        for template in self.templates.values():
            source, target = template.source, template.target
            graph.add_edge(source.identifier, target.identifier)
            if source.parent is not None:
                graph.add_edge(source.identifier, source.parent.identifier)
            if target.parent is not None:
                graph.add_edge(target.identifier, target.parent.identifier)
        return nx.is_tree(graph)  # Implementation note: nx tree is a poly-tree.

    def _cleanup(self) -> GenericTree:
        """
        Converts this builder's data into a cleaned up GenericTree.

        WARNING: The builder's internal data becomes invalid (and are therefore erased)
        after this method is called.
        """
        # Clean up templates by replacing variables with their parents.
        templates = list(self.templates.values())
        for template in templates:
            if template.source.parent is not None:
                template.source = template.source.parent
            if template.target.parent is not None:
                template.target = template.target.parent
        self.templates = self.case_links = None
        return GenericTree(templates)
