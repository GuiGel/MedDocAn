import logging

logging.basicConfig(level=logging.DEBUG)

import itertools as it
from io import StringIO
from typing import List
from unittest.mock import call, mock_open, patch

import pytest
from spacy.tokens import Doc

from meddocan.data import ArchiveFolder, meddocan_zip
from meddocan.data.containers import BratSpan, ExpandedEntity
from meddocan.data.docs_iterators import (
    DocWithBratPair,
    GsDocs,
    SysDocs,
    get_expanded_entities,
)
from tests.data.test_corpus import MockBratDocs

logger = logging.getLogger("meddocan")


class TestDocWithBratPair:
    def test(self, mock_docs):
        doc = (mock_docs[0].get_spacy_doc(),)
        brat_file_pair = next(meddocan_zip.brat_files(ArchiveFolder.train))
        doc_with_brat_pair = DocWithBratPair(brat_file_pair, doc)

        expected_stdout = (
            "DocWithBratPair(brat_files_pair=BratFilesPair("
            "ann=Path('/home/wave/.meddocan/datasets/meddocan/train-set.zip', "
            "'train/brat/S0004-06142005000500011-1.ann'), "
            "txt=Path('/home/wave/.meddocan/datasets/meddocan/train-set.zip', "
            "'train/brat/S0004-06142005000500011-1.txt')), "
            "doc=(Vivo en Aix en Provence! Soy Eric Laffont. ,))"
        )

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print(doc_with_brat_pair)
            assert mock_stdout.getvalue() == f"{expected_stdout}\n"


@pytest.mark.parametrize(
    argnames="brat_spans, expected",
    argvalues=(
        (
            [
                BratSpan(
                    id=None,
                    entity_type="NUM",
                    start=2,
                    end=5,
                    text="Aix en Provence",
                ),
                BratSpan(
                    id=None,
                    entity_type="PERS",
                    start=7,
                    end=9,
                    text="Eric Laffont",
                ),
            ],
            [],
        ),
        (
            [
                BratSpan(
                    id=None,
                    entity_type="LOC",
                    start=2,
                    end=5,
                    text="Aix en Provence",
                ),
                BratSpan(
                    id=None,
                    entity_type="PERS",
                    start=7,
                    end=8,
                    text="EricLaffont",
                ),
            ],
            [ExpandedEntity("EricLaffont", "Eric Laffont")],
        ),
    ),
)
def test_get_expanded_entities(
    mock_docs: List[Doc],
    brat_spans: List[BratSpan],
    expected: List[ExpandedEntity],
) -> None:
    """Test ``meddocan.data.docs_iterators.get_expanded_entities``.

    Args:
        doc (Doc): A spacy.tokens.Doc.
        brat_spans (List[BratSpan]): The entities that the doc must contains.
        expected (List[ExpandedEntity]): The expected results.
    """
    doc = mock_docs[0].get_spacy_doc()
    obtained = get_expanded_entities(doc, brat_spans)
    for obtained_ee, expected_ee in it.zip_longest(obtained, expected):
        assert obtained_ee == expected_ee


@pytest.mark.parametrize(
    argnames="brat_spans",
    argvalues=(
        [
            BratSpan(
                id=None,
                entity_type="LOC",
                start=2,
                end=5,
                text="Aix en Provence",
            ),
            BratSpan(
                id=None,
                entity_type="PERS",
                start=7,
                end=9,
                text="Eric Laffont",
            ),
        ],
    ),
)
def test_get_expanded_entities_bad_ents_num(
    mock_docs: List[Doc],
    brat_spans: List[BratSpan],
) -> None:
    """Test that ``meddocan.data.docs_iterators.get_expanded_entities`` is
    raising an :class:``AssertionError`` when the numbers of entities detected
    are not the same a the expected ones.

    Args:
        doc (Doc): A spacy.tokens.Doc.
        brat_spans (List[BratSpan]): The entities that the doc must contains.
        expected (List[ExpandedEntity]): The expected results.
    """
    doc = mock_docs[2].get_spacy_doc()
    with pytest.raises(AssertionError):
        get_expanded_entities(doc, brat_spans)


class TestGsDocs:
    """Class that group all the tests for the
    :class:`~meddocan.data.docs_iterators.GsDocs` object.
    """

    @pytest.mark.integration
    def test___iter__(self):
        gs_docs = GsDocs(ArchiveFolder.train)
        gs_doc = next(iter(gs_docs))

        expected_stdout = """DocWithBratPair(brat_files_pair=BratFilesPair\
(ann=Path('/home/wave/.meddocan/datasets/meddocan/train-set.zip', 'train/brat\
/S0004-06142005000500011-1.ann'), txt=Path('/home/wave/.meddocan/datasets/\
meddocan/train-set.zip', 'train/brat/S0004-06142005000500011-1.txt')), \
doc=Datos del paciente.
Nombre:  Ernesto.
Apellidos: Rivera Bueno.
NHC: 368503.
NASS: 26 63514095.
Domicilio:  Calle Miguel Benitez 90.
Localidad/ Provincia: Madrid.
CP: 28016.
Datos asistenciales.
Fecha de nacimiento: 03/03/1946.
País: España.
Edad: 70 años Sexo: H.
Fecha de Ingreso: 12/12/2016.
Médico:  Ignacio Navarro Cuéllar NºCol: 28 28 70973.
Informe clínico del paciente: Paciente de 70 años de edad, minero jubilado, \
sin alergias medicamentosas conocidas, que presenta como antecedentes \
personales: accidente laboral antiguo con fracturas vertebrales y costales; \
intervenido de enfermedad de Dupuytren en mano derecha y by-pass iliofemoral \
izquierdo; Diabetes Mellitus tipo II, hipercolesterolemia e hiperuricemia; \
enolismo activo, fumador de 20 cigarrillos / día.
Es derivado desde Atención Primaria por presentar hematuria macroscópica \
postmiccional en una ocasión y microhematuria persistente posteriormente, \
con micciones normales.
En la exploración física presenta un buen estado general, con abdomen y \
genitales normales; tacto rectal compatible con adenoma de próstata grado I/IV.
En la analítica de orina destaca la existencia de 4 hematíes/ campo y 0-5 \
leucocitos/campo; resto de sedimento normal.
Hemograma normal; en la bioquímica destaca una glucemia de 169 mg/dl y \
triglicéridos de 456 mg/dl; función hepática y renal normal. PSA de 1.16 ng/ml.
Las citologías de orina son repetidamente sospechosas de malignidad.
En la placa simple de abdomen se valoran cambios degenerativos en columna \
lumbar y calcificaciones vasculares en ambos hipocondrios y en pelvis.
La ecografía urológica pone de manifiesto la existencia de quistes corticales \
simples en riñón derecho, vejiga sin alteraciones con buena capacidad y \
próstata con un peso de 30 g.
En la UIV se observa normofuncionalismo renal bilateral, calcificaciones \
sobre silueta renal derecha y uréteres arrosariados con imágenes de adición \
en el tercio superior de ambos uréteres, en relación a pseudodiverticulosis \
ureteral. El cistograma demuestra una vejiga con buena capacidad, pero \
paredes trabeculadas en relación a vejiga de esfuerzo. La TC abdominal es \
normal.
La cistoscopia descubre la existencia de pequeñas tumoraciones vesicales, \
realizándose resección transuretral con el resultado anatomopatológico de \
carcinoma urotelial superficial de vejiga.
Remitido por: Ignacio Navarro Cuéllar c/ del Abedul 5-7, 2º dcha 28036 \
Madrid, España E-mail: nnavcu@hotmail.com.
)"""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print(gs_doc)
            assert mock_stdout.getvalue() == f"{expected_stdout}\n"

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "write_sentences, document_separator_token",
        [(True, None)],
    )
    def test_to_connl03(
        self,
        mock_docs: List[Doc],
        write_sentences: bool,
        document_separator_token: str,
    ) -> None:

        # 1. Use the mock_docs as output of the __iter__ method
        # of the MockBratDocs class.

        with patch.object(
            GsDocs,
            "__iter__",
            new_callable=lambda: MockBratDocs(mock_docs[:2]),
        ):
            gs_docs = GsDocs(ArchiveFolder.train)
            m = mock_open()
            with patch("pathlib.Path.open", m):
                gs_docs.to_connl03(
                    file="fake-file.txt",
                    write_sentences=write_sentences,
                    document_separator_token=document_separator_token,
                )
            # 2. Verify that the calls to the Path object are the expected
            # ones.

            assert m.mock_calls == [
                call(mode="w", encoding="utf-8", newline="\n"),
                call().__enter__(),
                call().writelines(
                    [
                        "Vivo O\n",
                        "en O\n",
                        "Aix B-LOC\n",
                        "en I-LOC\n",
                        "Provence I-LOC\n",
                        "! O\n",
                        "\n",
                        "Soy O\n",
                        "Eric B-PERS\n",
                        "Laffont I-PERS\n",
                        ". O\n",
                        "\n",
                    ]
                ),
                call().__exit__(None, None, None),
                call(mode="a", encoding="utf-8", newline="\n"),
                call().__enter__(),
                call().writelines(
                    [
                        "Vivo O\n",
                        "en O\n",
                        "Bilbao B-LOC\n",
                        "! O\n",
                        "\n",
                        "Soy O\n",
                        "Zaira B-PERS\n",
                        "Aurrekoetxea I-PERS\n",
                        ". O\n",
                        "\n",
                    ]
                ),
                call().__exit__(None, None, None),
            ]


class TestSysDocs:
    """Class that group all the tests for the
    :class:`~meddocan.data.docs_iterators.SysDocs` object.
    """

    @pytest.mark.integration
    def test___iter__(self):
        sys_docs = SysDocs(ArchiveFolder.train, model=None)
        sys_doc = next(iter(sys_docs))
        expected_stdout = """DocWithBratPair(brat_files_pair=BratFilesPair\
(ann=Path('/home/wave/.meddocan/datasets/meddocan/train-set.zip', 'train/brat\
/S0004-06142005000500011-1.ann'), txt=Path('/home/wave/.meddocan/datasets/\
meddocan/train-set.zip', 'train/brat/S0004-06142005000500011-1.txt')), \
doc=Datos del paciente.
Nombre:  Ernesto.
Apellidos: Rivera Bueno.
NHC: 368503.
NASS: 26 63514095.
Domicilio:  Calle Miguel Benitez 90.
Localidad/ Provincia: Madrid.
CP: 28016.
Datos asistenciales.
Fecha de nacimiento: 03/03/1946.
País: España.
Edad: 70 años Sexo: H.
Fecha de Ingreso: 12/12/2016.
Médico:  Ignacio Navarro Cuéllar NºCol: 28 28 70973.
Informe clínico del paciente: Paciente de 70 años de edad, minero jubilado, \
sin alergias medicamentosas conocidas, que presenta como antecedentes \
personales: accidente laboral antiguo con fracturas vertebrales y costales; \
intervenido de enfermedad de Dupuytren en mano derecha y by-pass iliofemoral \
izquierdo; Diabetes Mellitus tipo II, hipercolesterolemia e hiperuricemia; \
enolismo activo, fumador de 20 cigarrillos / día.
Es derivado desde Atención Primaria por presentar hematuria macroscópica \
postmiccional en una ocasión y microhematuria persistente posteriormente, \
con micciones normales.
En la exploración física presenta un buen estado general, con abdomen y \
genitales normales; tacto rectal compatible con adenoma de próstata grado I/IV.
En la analítica de orina destaca la existencia de 4 hematíes/ campo y 0-5 \
leucocitos/campo; resto de sedimento normal.
Hemograma normal; en la bioquímica destaca una glucemia de 169 mg/dl y \
triglicéridos de 456 mg/dl; función hepática y renal normal. PSA de 1.16 ng/ml.
Las citologías de orina son repetidamente sospechosas de malignidad.
En la placa simple de abdomen se valoran cambios degenerativos en columna \
lumbar y calcificaciones vasculares en ambos hipocondrios y en pelvis.
La ecografía urológica pone de manifiesto la existencia de quistes corticales \
simples en riñón derecho, vejiga sin alteraciones con buena capacidad y \
próstata con un peso de 30 g.
En la UIV se observa normofuncionalismo renal bilateral, calcificaciones \
sobre silueta renal derecha y uréteres arrosariados con imágenes de adición \
en el tercio superior de ambos uréteres, en relación a pseudodiverticulosis \
ureteral. El cistograma demuestra una vejiga con buena capacidad, pero \
paredes trabeculadas en relación a vejiga de esfuerzo. La TC abdominal es \
normal.
La cistoscopia descubre la existencia de pequeñas tumoraciones vesicales, \
realizándose resección transuretral con el resultado anatomopatológico de \
carcinoma urotelial superficial de vejiga.
Remitido por: Ignacio Navarro Cuéllar c/ del Abedul 5-7, 2º dcha 28036 \
Madrid, España E-mail: nnavcu@hotmail.com.
)"""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print(sys_doc)
            assert mock_stdout.getvalue() == f"{expected_stdout}\n"

        assert sys_doc.doc._.is_meddocan_doc
