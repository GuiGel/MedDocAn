# Introducción

Nos interesaremos por los métodos de Transfer Learning basados en los transformadores. Para ello, participaremos ficticiamente en el concurso **MEDDOCAN** (tarea 9 del IberLEF 2019 [^1]) al que el grupo de semántica de Serikat ha sido invitado en 2019 por el Pan TL (El Plan de Impulso de las Tecnologías del Lenguaje) [^2].

Este concurso se ha creado porque la anonimización de la información sensible para la privacidad tiene una importancia creciente en la era de la digitalización y el aprendizaje automático. Es, en particular, relevante para los textos del ámbito médico que contienen un gran número de información sensible por naturaleza. La tarea compartida **MEDDOCAN** (Medical Document Anonymization) {cite}`Marimon2019AutomaticDO` tiene como objetivo detectar automáticamente la información sanitaria protegida (PHI) de los documentos médicos españoles.  Tras la pasada tarea de desidentificación de los resúmenes de PubMed en inglés {cite}`Stubbs2015-fg`, fue el primer concurso sobre este tema en datos españoles.


```{note}
La amplia mayoría de las tablas, figuras y números facilitados en este documento son producidos gracias al uso de Jupyter Notebooks [^3]. Se almacenan los resultados del código e se insertan en nuestro informe de tal mañera que se puede averiguar la veracidad de los resultados mirando el código.
```

[^1]: http://ceur-ws.org/Vol-2421/
[^2]: https://plantl.mineco.gob.es/Paginas/index.aspx 
[^3]: https://jupyter.org/