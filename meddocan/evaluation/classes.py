import abc
import os
from typing import Any, Dict, List, NamedTuple, Optional, Set, TypeVar, Union
from xml.etree import ElementTree

from meddocan.evaluation.tags import PHITag


class Ner(NamedTuple):
    tag: str
    start: int
    end: int


class Span(NamedTuple):
    start: int
    end: int


T = TypeVar("T", Ner, PHITag)


class Annotation(abc.ABC):
    def __init__(self, file_name: str, root: str = "root") -> None:
        self.text: Optional[str] = None
        self.num_sentences: Optional[int] = None
        self.root = root
        self.phi: List[Ner] = []
        self.sensitive_spans: List[Span] = []
        self.sensitive_spans_merged: List[Span] = []
        self.verbose: bool = False

        self.sys_id = os.path.basename(os.path.dirname(file_name))
        self.doc_id = os.path.splitext(os.path.basename(file_name))[0]

        self.parse_text_and_tags(file_name)
        self.parse_text_and_spans(file_name)
        self.file_name = file_name

    @property
    def id(self) -> str:
        return self.doc_id

    def get_phi(self) -> List[Ner]:
        return self.phi

    def get_phi_spans(self) -> List[Span]:
        return self.sensitive_spans

    def get_phi_spans_merged(self) -> List[Span]:
        return self.sensitive_spans_merged

    def get_number_sentences(self) -> Optional[int]:
        if self.doc_id is not None:
            try:
                self.num_sentences = sum(
                    1
                    for line in open(
                        "annotated_corpora/sentence_splitted/"
                        + self.doc_id
                        + ".ann"
                    )
                )
            except IOError:
                print(
                    "File '"
                    + "freeling/sentence_splitted/"
                    + self.doc_id
                    + ".ann' not found."
                )
        return self.num_sentences

    def add_spans(self, phi_tags: List[Span]) -> None:
        # Increment `Annotation.sensitive_spans` attribute and
        # `Annotation.sensitive_spans_merged`.

        for tag in sorted(phi_tags):
            self.sensitive_spans.append(tag)

        for y in sorted(phi_tags):
            if not self.sensitive_spans_merged:
                self.sensitive_spans_merged.append(y)
            else:
                x = self.sensitive_spans_merged.pop()
                # If phi_tags is not empty text can't be None
                assert self.text is not None
                if self.is_all_non_alphanumeric(self.text[x[1] : y[0]]):
                    self.sensitive_spans_merged.append(Span(x[0], y[1]))
                else:
                    self.sensitive_spans_merged.append(x)
                    self.sensitive_spans_merged.append(y)

    @staticmethod
    def is_all_non_alphanumeric(string: str) -> bool:
        for i in string:
            if i.isalnum():
                return False
        return True

    @abc.abstractmethod
    def parse_text_and_tags(self, file_name: str) -> None:
        """_summary_

        Args:
            file_name (str): Name of the file to parse
        """

    @abc.abstractmethod
    def parse_text_and_spans(self, file_name: str) -> None:
        """_summary_

        Args:
            file_name (str): Name of the file to parse
        """


class i2b2Annotation(Annotation):
    """This class models the i2b2 annotation format."""

    def parse_text_and_tags(self, file_name: str) -> None:
        text = open(file_name, "r").read()
        self.text = text

        tree = ElementTree.parse(file_name)
        root = tree.getroot()

        self.root = root.tag

        if (element := root.find("TEXT")) is not None:
            self.text = element.text
        else:
            self.text = None

        # Handles files where PHI, and AnnotatorTags are all just
        # stuffed into tag element.
        for t, cls in PHITag.tag_types.items():
            if (tags := root.find("TAGS")) is not None:
                if len(tags.findall(t)):
                    for element in tags.findall(t):
                        # self.phi.append(cls(element))
                        tag = cls(element)
                        self.phi.append(
                            Ner(tag.name, tag.get_start(), tag.get_end())
                        )

    def parse_text_and_spans(self, file_name: str) -> None:
        text = open(file_name, "r").read()
        self.text = text

        tree = ElementTree.parse(file_name)
        root = tree.getroot()

        self.root = root.tag

        if (element := root.find("TEXT")) is not None:
            self.text = element.text
        else:
            self.text = None

        # Fill list with tuples (start, end) for each annotation
        phi_tags: List[Span] = []
        for t, cls in PHITag.tag_types.items():
            if (tags := root.find("TAGS")) is not None:
                if elements := tags.findall(t):
                    for element in elements:
                        phi_tags.append(
                            Span(
                                cls(element).get_start(),
                                cls(element).get_end(),
                            )
                        )

        # Store spans
        self.add_spans(phi_tags)


class BratAnnotation(Annotation):
    """This class models the BRAT annotation format."""

    def parse_text_and_tags(self, file_name: str) -> None:
        text = open(os.path.splitext(file_name)[0] + ".txt", "r").read()
        self.text = text

        for row in open(file_name, "r"):
            line = row.strip()  # Ex: 'T1\tLabel Start End\ttext'
            if line.startswith("T"):  # Lines is a Brat TAG
                try:
                    label = line.split("\t")[1].split()
                    tag = label[0]
                    start = int(label[1])
                    end = int(label[2])
                    self.phi.append(Ner(tag, start, end))
                except IndexError:
                    print(
                        "ERROR! Index error while splitting sentence '"
                        + line
                        + "' in document '"
                        + file_name
                        + "'!"
                    )
            else:  # Line is a Brat comment
                if self.verbose:
                    print("\tSkipping line (comment):\t" + line)

    def parse_text_and_spans(self, file_name: str) -> None:
        text = open(os.path.splitext(file_name)[0] + ".txt", "r").read()
        self.text = text

        phi_tags: List[Span] = []
        for row in open(file_name, "r"):
            line = row.strip()  # Ex: 'T1\tLabel Start End\ttext'
            if line.startswith("T"):  # Lines is a Brat TAG
                try:
                    label = line.split("\t")[1].split()
                    start = int(label[1])
                    end = int(label[2])

                    phi_tags.append(Span(start, end))
                except IndexError:
                    print(
                        "ERROR! Index error while splitting sentence '"
                        + line
                        + "' in document '"
                        + file_name
                        + "'!"
                    )
            else:  # Line is a Brat comment
                if self.verbose:
                    print("\tSkipping line (comment):\t" + line)

        # Store spans
        self.add_spans(phi_tags)


class Evaluate:
    """Base class with all methods to evaluate the different subtracks."""

    def __init__(
        self,
        sys_ann: Dict[str, Annotation],
        gs_ann: Dict[str, Annotation],
    ) -> None:
        # https://stackoverflow.com/questions/58906541/incompatible-types-in-assignment-expression-has-type-listnothing-variabl
        self.tp: Union[List[Set[Ner]], List[Set[Span]]] = []  # type: ignore[assignment]
        self.fp: Union[List[Set[Ner]], List[Set[Span]]] = []  # type: ignore[assignment]
        self.fn: Union[List[Set[Ner]], List[Set[Span]]] = []  # type: ignore[assignment]
        self.doc_ids: List[str] = []
        self.verbose = False

        self.sys_id = sys_ann[list(sys_ann.keys())[0]].sys_id

        self.label: str = ""

    @staticmethod
    def get_tagset_ner(
        annotation: Annotation,
    ) -> List[Ner]:
        return annotation.get_phi()

    @staticmethod
    def get_tagset_span(
        annotation: Annotation,
    ) -> List[Span]:
        return annotation.get_phi_spans()

    @staticmethod
    def get_tagset_span_merged(
        annotation: Annotation,
    ) -> List[Span]:
        return annotation.get_phi_spans_merged()

    @staticmethod
    def is_contained(content: Span, container: Set[Span]) -> bool:
        """Verify that the content Span is contained in the Set of Spans
        container.

        Example:

        >>> Evaluate.is_contained(Span(10, 20), {Span(10, 20), Span(33, 45)})
        True

        Args:
            content (Span): Span to check.
            container (Set[Span]): Set of Spans.

        Returns:
            bool: True is Span is included or the same as one of the container
                elements.
        """
        for element in sorted(container):
            if content[0] >= element[0] and content[1] <= element[1]:
                return True
        return False

    @staticmethod
    def recall(
        tp: Union[Set[Ner], Set[Span]], fn: Union[Set[Ner], Set[Span]]
    ) -> float:
        try:
            return len(tp) / float(len(fn) + len(tp))
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def precision(
        tp: Union[Set[Ner], Set[Span]], fp: Union[Set[Ner], Set[Span]]
    ) -> float:
        try:
            return len(tp) / float(len(fp) + len(tp))
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def F_beta(p: float, r: float, beta: float = 1) -> float:
        try:
            return (1 + beta**2) * ((p * r) / (p + r))
        except ZeroDivisionError:
            return 0.0

    def micro_recall(self) -> float:
        try:
            return sum([len(t) for t in self.tp]) / float(
                sum([len(t) for t in self.tp]) + sum([len(t) for t in self.fn])
            )
        except ZeroDivisionError:
            return 0.0

    def micro_precision(self) -> float:
        try:
            return sum([len(t) for t in self.tp]) / float(
                sum([len(t) for t in self.tp]) + sum([len(t) for t in self.fp])
            )
        except ZeroDivisionError:
            return 0.0

    def _print_docs(self) -> None:
        for i, doc_id in enumerate(self.doc_ids):
            mp = Evaluate.precision(self.tp[i], self.fp[i])
            mr = Evaluate.recall(self.tp[i], self.fn[i])

            str_fmt = "{:<35}{:<15}{:<20}"

            print(str_fmt.format(doc_id, "Precision", "{:.4}".format(mp)))

            print(str_fmt.format("", "Recall", "{:.4}".format(mr)))

            print(
                str_fmt.format(
                    "", "F1", "{:.4}".format(Evaluate.F_beta(mp, mr))
                )
            )

            print("{:-<60}".format(""))

    def _print_summary(self) -> None:
        mp = self.micro_precision()
        mr = self.micro_recall()

        str_fmt = "{:<35}{:<15}{:<20}"

        print(str_fmt.format("", "", ""))

        print("Report (" + self.sys_id + "):")

        print("{:-<60}".format(""))

        print(str_fmt.format(self.label, "Measure", "Micro"))

        print("{:-<60}".format(""))

        print(
            str_fmt.format(
                "Total ({} docs)".format(len(self.doc_ids)),
                "Precision",
                "{:.4}".format(mp),
            )
        )

        print(str_fmt.format("", "Recall", "{:.4}".format(mr)))

        print(
            str_fmt.format("", "F1", "{:.4}".format(Evaluate.F_beta(mr, mp)))
        )

        print("{:-<60}".format(""))

        print("\n")

    def print_docs(self) -> None:
        print("\n")
        print("Report ({}):".format(self.sys_id))
        print("{:-<60}".format(""))
        print("{:<35}{:<15}{:<20}".format("Document ID", "Measure", "Micro"))
        print("{:-<60}".format(""))
        self._print_docs()

    def print_report(self, verbose: bool = False) -> None:
        self.verbose = verbose
        if verbose:
            self.print_docs()

        self._print_summary()


class EvaluateSubtrack1(Evaluate):
    """Class for running the NER evaluation."""

    def __init__(
        self,
        sys_sas: Dict[str, Annotation],
        gs_sas: Dict[str, Annotation],
    ) -> None:
        self.tp: List[Set[Ner]] = []
        self.fp: List[Set[Ner]] = []
        self.fn: List[Set[Ner]] = []
        self.num_sentences: List[Optional[int]] = []
        self.doc_ids: List[str] = []
        self.verbose = False

        self.sys_id = sys_sas[list(sys_sas.keys())[0]].sys_id

        self.label = "Subtrack 1 [NER]"

        for doc_id in sorted(list(set(sys_sas.keys()) & set(gs_sas.keys()))):

            gold = set(self.get_tagset_ner(gs_sas[doc_id]))
            sys = set(self.get_tagset_ner(sys_sas[doc_id]))
            num_sentences = self.get_num_sentences(sys_sas[doc_id])

            self.tp.append(gold.intersection(sys))
            self.fp.append(sys - gold)
            self.fn.append(gold - sys)
            self.num_sentences.append(num_sentences)
            self.doc_ids.append(doc_id)

    @staticmethod
    def get_num_sentences(annotation: Annotation) -> Optional[int]:
        return annotation.get_number_sentences()

    @staticmethod
    def leak_score(
        fn: Set[Ner], num_sentences: Optional[int]
    ) -> Union[float, str]:
        if num_sentences is not None:
            if num_sentences:
                return float(len(fn) / num_sentences)
            else:
                return 0.0
        else:
            return "NA"

    def micro_leak(self) -> Union[float, str]:
        num_sentences = (t for t in self.num_sentences if t is not None)
        if any(1 for t in num_sentences):
            # At least one element is not None
            try:
                return float(
                    sum(len(t) for t in self.fn)
                    / sum(t for t in num_sentences)
                )
            except ZeroDivisionError:
                return 0.0
        else:
            return "NA"

    def _print_docs(self) -> None:
        for i, doc_id in enumerate(self.doc_ids):
            mp = EvaluateSubtrack1.precision(self.tp[i], self.fp[i])
            mr = EvaluateSubtrack1.recall(self.tp[i], self.fn[i])
            leak = EvaluateSubtrack1.leak_score(
                self.fn[i], self.num_sentences[i]
            )

            str_fmt = "{:<35}{:<15}{:<20}"

            print(str_fmt.format(doc_id, "Leak", "{:.4}".format(leak)))

            print(str_fmt.format("", "Precision", "{:.4}".format(mp)))

            print(str_fmt.format("", "Recall", "{:.4}".format(mr)))

            print(
                str_fmt.format(
                    "", "F1", "{:.4}".format(Evaluate.F_beta(mp, mr))
                )
            )

            print("{:-<60}".format(""))

    def _print_summary(self) -> None:
        mp = self.micro_precision()
        mr = self.micro_recall()
        ml = self.micro_leak()

        str_fmt = "{:<35}{:<15}{:<20}"

        print(str_fmt.format("", "", ""))

        print("Report (" + self.sys_id + "):")

        print("{:-<60}".format(""))

        print(str_fmt.format(self.label, "Measure", "Micro"))

        print("{:-<60}".format(""))

        print(
            str_fmt.format(
                "Total ({} docs)".format(len(self.doc_ids)),
                "Leak",
                "{:.4}".format(ml),
            )
        )

        print(str_fmt.format("", "Precision", "{:.4}".format(mp)))

        print(str_fmt.format("", "Recall", "{:.4}".format(mr)))

        print(
            str_fmt.format("", "F1", "{:.4}".format(Evaluate.F_beta(mr, mp)))
        )

        print("{:-<60}".format(""))

        print("\n")


class EvaluateSubtrack2(Evaluate):
    """Class for running the SPAN evaluation with strict span mode."""

    def __init__(
        self,
        sys_sas: Dict[str, Annotation],
        gs_sas: Dict[str, Annotation],
    ):
        self.tp: List[Set[Span]] = []
        self.fp: List[Set[Span]] = []
        self.fn: List[Set[Span]] = []
        self.doc_ids: List[str] = []
        self.verbose: bool = False

        self.sys_id = sys_sas[list(sys_sas.keys())[0]].sys_id
        self.label = "Subtrack 2 [strict]"

        for doc_id in sorted(list(set(sys_sas.keys()) & set(gs_sas.keys()))):

            gold = set(self.get_tagset_span(gs_sas[doc_id]))
            sys = set(self.get_tagset_span(sys_sas[doc_id]))

            self.tp.append(gold.intersection(sys))
            self.fp.append(sys - gold)
            self.fn.append(gold - sys)
            self.doc_ids.append(doc_id)


class EvaluateSubtrack2merged(Evaluate):
    """Class for running the SPAN evaluation with merged spans mode."""

    def __init__(
        self,
        sys_sas: Dict[str, Annotation],
        gs_sas: Dict[str, Annotation],
    ) -> None:
        self.tp: List[Set[Span]] = []
        self.fp: List[Set[Span]] = []
        self.fn: List[Set[Span]] = []
        self.doc_ids: List[str] = []
        self.verbose: bool = False

        self.sys_id = sys_sas[list(sys_sas.keys())[0]].sys_id
        self.label = "Subtrack 2 [merged]"

        for doc_id in sorted(list(set(sys_sas.keys()) & set(gs_sas.keys()))):

            gold_strict = set(self.get_tagset_span(gs_sas[doc_id]))
            sys_strict = set(self.get_tagset_span(sys_sas[doc_id]))

            gold_merged = set(self.get_tagset_span_merged(gs_sas[doc_id]))
            sys_merged = set(self.get_tagset_span_merged(sys_sas[doc_id]))

            intersection = gold_strict.intersection(sys_strict).union(
                gold_merged.intersection(sys_merged)
            )

            fp = sys_strict - gold_strict
            for tag in sys_strict:
                if self.is_contained(tag, intersection):
                    if tag in fp:
                        fp.remove(tag)

            fn = gold_strict - sys_strict
            for tag in gold_strict:
                if self.is_contained(tag, intersection):
                    if tag in fn:
                        fn.remove(tag)

            self.tp.append(intersection)
            self.fp.append(fp)
            self.fn.append(fn)
            self.doc_ids.append(doc_id)


class MeddocanEvaluation:
    """Base class for running the evaluations."""

    def __init__(self) -> None:
        self.evaluations: List[Evaluate] = []

    def add_eval(self, e: Evaluate, label: str = "") -> None:
        e.sys_id = "SYSTEM: " + e.sys_id
        e.label = label
        self.evaluations.append(e)

    def print_docs(self) -> None:
        for e in self.evaluations:
            e.print_docs()

    def print_report(self, verbose: bool = False) -> None:
        for e in self.evaluations:
            e.print_report(verbose=verbose)


class NER_Evaluation(MeddocanEvaluation):
    """Class for running the NER evaluation (Subtrack 1)."""

    def __init__(
        self,
        annotator_cas: Dict[str, Annotation],
        gold_cas: Dict[str, Annotation],
        **kwargs: Dict[str, Any],
    ) -> None:
        self.evaluations = []

        # Basic Evaluation
        self.add_eval(
            EvaluateSubtrack1(annotator_cas, gold_cas, **kwargs),
            label="SubTrack 1 [NER]",
        )


class Span_Evaluation(MeddocanEvaluation):
    """Class for running the SPAN evaluation (Subtrack 2). Calls to 'strict'
    and 'merged' evaluations."""

    def __init__(
        self,
        annotator_cas: Dict[str, Annotation],
        gold_cas: Dict[str, Annotation],
        **kwargs: Dict[str, Any],
    ):
        self.evaluations = []

        self.add_eval(
            EvaluateSubtrack2(annotator_cas, gold_cas, **kwargs),
            label="SubTrack 2 [strict]",
        )

        self.add_eval(
            EvaluateSubtrack2merged(annotator_cas, gold_cas, **kwargs),
            label="SubTrack 2 [merged]",
        )
