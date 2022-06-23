###############################################################################
#
#    This file manages all tag related to i2b2 classes and encapsulates much
#    of the functionality that the rest of this evaluation script works off of.
#
#    Class hierarchy reference
#
#    [+]Tag
#       [+]AnnotatorTag
#          [+]PHITag
#             [+]NameTag
#             [+]ProfessionTag
#             [+]LocationTag
#             [+]AgeTag
#             [+]DateTag
#             [+]ContactTag
#             [+]IDTag
#             [+]OtherTag

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple, Type
from xml.etree.ElementTree import Element


class Tag:
    """Base Tag object"""

    attributes: Dict[str, Callable[[str], bool]] = OrderedDict()
    key: List[str] = []

    def __init__(self, element: Element) -> None:
        self.name = element.tag
        try:
            self.id: str = element.attrib["id"]
        except KeyError:
            self.id = ""

    def _get_key(self) -> Tuple[str, ...]:
        key = []
        for k in self.key:
            key.append(getattr(self, k))
        return tuple(key)

    def _key_equality(self, other: Tag) -> bool:
        return (
            self._get_key() == other._get_key()
            and other._get_key() == self._get_key()
        )

    def _key_hash(self) -> int:
        return hash(self._get_key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tag):
            return NotImplemented
        return self._key_equality(other)

    def __hash__(self) -> int:
        return self._key_hash()


def isint(x: Any) -> bool:
    try:
        int(x)
        return True
    except ValueError:
        return False


class AnnotatorTag(Tag):
    """Defines the tags that model the general tags produced by annotators.
    AnnotatorTag also implements the functions that convert an annotator tag
    to a DocumentTag - a tag designed to annotate document level information
    rather than specific positional information.
    """

    attributes: Dict[str, Callable[[str], bool]] = OrderedDict()
    attributes["id"] = lambda v: True
    attributes["docid"] = lambda v: True
    attributes["start"] = isint
    attributes["end"] = isint
    attributes["text"] = lambda v: True

    key = ["name"]

    def __init__(self, element: Element):
        super(AnnotatorTag, self).__init__(element)

        self.start: int
        self.end: int

        for k, validp in self.attributes.items():
            if k in element.attrib.keys():
                if validp(element.attrib[k]):
                    setattr(self, k, element.attrib[k])
                else:
                    fstr = "WARNING: Expected attribute '{}' for xml element "
                    fstr += "<{} ({})>  was not valid ('{}')"
                    print(
                        fstr.format(
                            k,
                            element.tag,
                            element.attrib["id"],
                            element.attrib[k],
                        )
                    )
                    setattr(self, k, element.attrib[k])

            elif k in self.key:
                fstr = "WARNING: Expected attribute '%s' for xml element "
                fstr += "<%s ('%s')>, setting to ''"
                print(fstr.format(k, element.tag, element.attrib["id"]))

                setattr(self, k, "")

    def get_start(self) -> int:
        try:
            return int(self.start)
        except TypeError:
            return self.start

    def get_end(self) -> int:
        try:
            return int(self.end)
        except TypeError:
            return self.end


class PHITag(AnnotatorTag):
    valid_TYPE = [
        "NOMBRE_SUJETO_ASISTENCIA",
        "NOMBRE_PERSONAL_SANITARIO",
        "PROFESION",
        "HOSPITAL",
        "INSTITUCION",
        "CALLE",
        "TERRITORIO",
        "PAIS",
        "CENTRO_SALUD",
        "EDAD_SUJETO_ASISTENCIA",
        "FECHAS",
        "NUMERO_TELEFONO",
        "NUMERO_FAX",
        "CORREO_ELECTRONICO",
        "URL_WEB",
        "ID_ASEGURAMIENTO",
        "ID_CONTACTO_ASISTENCIAL",
        "NUMERO_BENEF_PLAN_SALUD",
        "IDENTIF_VEHICULOS_NRSERIE_PLACAS",
        "IDENTIF_DISPOSITIVOS_NRSERIE",
        "IDENTIF_BIOMETRICOS",
        "ID_SUJETO_ASISTENCIA",
        "ID_TITULACION_PERSONAL_SANITARIO",
        "ID_EMPLEO_PERSONAL_SANITARIO",
        "OTRO_NUMERO_IDENTIF",
        "SEXO_SUJETO_ASISTENCIA",
        "FAMILIARES_SUJETO_ASISTENCIA",
        "OTROS_SUJETO_ASISTENCIA",
        "DIREC_PROT_INTERNET",
        "TOKEN",
    ]

    attributes = OrderedDict(AnnotatorTag.attributes.items())
    attributes["TYPE"] = lambda v: v in PHITag.valid_TYPE

    key = AnnotatorTag.key + ["start", "end", "TYPE"]

    tag_types: Dict[str, Type[PHITag]]


class NameTag(PHITag):
    valid_TYPE = ["NOMBRE_SUJETO_ASISTENCIA", "NOMBRE_PERSONAL_SANITARIO"]
    attributes = OrderedDict(PHITag.attributes.items())
    attributes["TYPE"] = lambda v: v in NameTag.valid_TYPE


class ProfessionTag(PHITag):
    valid_TYPE = ["PROFESION"]
    attributes = OrderedDict(PHITag.attributes.items())
    attributes["TYPE"] = lambda v: v in ProfessionTag.valid_TYPE


class LocationTag(PHITag):
    valid_TYPE = [
        "HOSPITAL",
        "INSTITUCION",
        "CALLE",
        "TERRITORIO",
        "PAIS",
        "CENTRO_SALUD",
    ]
    attributes = OrderedDict(PHITag.attributes.items())
    attributes["TYPE"] = lambda v: v in LocationTag.valid_TYPE


class AgeTag(PHITag):
    valid_TYPE = ["EDAD_SUJETO_ASISTENCIA"]
    attributes = OrderedDict(PHITag.attributes.items())
    attributes["TYPE"] = lambda v: v in AgeTag.valid_TYPE


class DateTag(PHITag):
    valid_TYPE = ["FECHAS"]
    attributes = OrderedDict(PHITag.attributes.items())
    attributes["TYPE"] = lambda v: v in DateTag.valid_TYPE


class ContactTag(PHITag):
    valid_TYPE = [
        "NUMERO_TELEFONO",
        "NUMERO_FAX",
        "CORREO_ELECTRONICO",
        "URL_WEB",
    ]
    attributes = OrderedDict(PHITag.attributes.items())
    attributes["TYPE"] = lambda v: v in ContactTag.valid_TYPE


class IDTag(PHITag):
    valid_TYPE = [
        "ID_ASEGURAMIENTO",
        "ID_CONTACTO_ASISTENCIAL",
        "NUMERO_BENEF_PLAN_SALUD",
        "IDENTIF_VEHICULOS_NRSERIE_PLACAS",
        "IDENTIF_DISPOSITIVOS_NRSERIE",
        "IDENTIF_BIOMETRICOS",
        "ID_SUJETO_ASISTENCIA",
        "ID_TITULACION_PERSONAL_SANITARIO",
        "ID_EMPLEO_PERSONAL_SANITARIO",
        "OTRO_NUMERO_IDENTIF",
    ]
    attributes = OrderedDict(PHITag.attributes.items())
    attributes["TYPE"] = lambda v: v in IDTag.valid_TYPE


class OtherTag(PHITag):
    valid_TYPE = [
        "SEXO_SUJETO_ASISTENCIA",
        "FAMILIARES_SUJETO_ASISTENCIA",
        "OTROS_SUJETO_ASISTENCIA",
        "DIREC_PROT_INTERNET",
    ]
    attributes = OrderedDict(PHITag.attributes.items())
    attributes["TYPE"] = lambda v: v in OtherTag.valid_TYPE


setattr(
    PHITag,
    "tag_types",
    {
        "PHI": PHITag,
        "NAME": NameTag,
        "PROFESSION": ProfessionTag,
        "LOCATION": LocationTag,
        "AGE": AgeTag,
        "DATE": DateTag,
        "CONTACT": ContactTag,
        "ID": IDTag,
        "OTHER": OtherTag,
    },
)


PHI_TAG_CLASSES = [
    NameTag,
    ProfessionTag,
    LocationTag,
    AgeTag,
    DateTag,
    ContactTag,
    IDTag,
    OtherTag,
]


# Comment should be last in tag order,  so add it down here
# that way all other sub tags have had their attributes set first
# This also provides the MEDICAL_TAG_CLASSES list.
for c in PHI_TAG_CLASSES:
    c.attributes["comment"] = lambda v: True
