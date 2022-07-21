"""Module to test class of the `meddocan.data.containers` module."""
import functools

from meddocan.data.containers import BratAnnotations, BratSpan


class TestBratSpan:
    def test_entity(self) -> None:
        brat_span = BratSpan(
            id=None, entity_type="PERS", start=2, end=5, text="Aix en Provence"
        )
        assert brat_span.entity == (2, 5, "PERS")

    def test_from_bytes(self) -> None:
        line = "T1\tCALLE 2365 2391\tc/ del Abedul 5-7, 2º dcha\n"
        byte_line = line.encode("utf-8")
        obtained = BratSpan.from_bytes(byte_line)
        expected = BratSpan(
            id="T1",
            entity_type="CALLE",
            start=2365,
            end=2391,
            text="c/ del Abedul 5-7, 2º dcha",
        )
        assert obtained == expected, f"{obtained} != {expected}"


class TestBratAnnotations:

    lines = "Vivo en Bilbao\nMe llaman Arizio."
    brat_spans = [
        BratSpan(id=None, entity_type="LOC", start=8, end=14, text="Bilbao")
    ]

    @functools.lru_cache
    def get_brat_annotations(cls):
        # Could be done by a fixture also...
        return BratAnnotations(cls.lines, cls.brat_spans)

    def test___len__(self):
        assert len(self.get_brat_annotations()) == 1
