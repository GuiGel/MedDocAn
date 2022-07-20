# MEDDOCAN: Medical Document Anonymization Track


<p align="center">
<a href="https://github.com/GuiGel/MedDocAn/actions?query=workflow%3ATest" target="_blank">
    <img src="https://github.com/GuiGel/MedDocAn/workflows/Test/badge.svg" alt="Test">
</a>
<a href="https://codecov.io/gh/GuiGel/MedDocAn" target="_blank">
    <img src="https://img.shields.io/codecov/c/github/GuiGel/MedDocAn/branch/dev?color=%2334D058" alt="Coverage">
</a>

*Our results on the Medical Document Anonymization Track (track 9 of the <abbr title="Iberian Languages Evaluation Forum 2019"> [IberLEF 2019](http://ceur-ws.org/Vol-2421/)).</abbr>*

## About the task

Clinical records with protected health information (PHI) cannot be directly shared “as is”, due to privacy constraints, making it particularly cumbersome to carry out NLP research in the medical domain. A necessary precondition for accessing clinical records outside of hospitals is their de-identification, i.e., the exhaustive removal, or replacement, of all mentioned PHI phrases.

The practical relevance of anonymization or de-identification of clinical texts motivated the proposal of two shared tasks, the 2006 and 2014 de-identification tracks, organized under the umbrella of the i2b2 (i2b2.org) community evaluation effort. The i2b2 effort has deeply influenced the clinical NLP community worldwide, but was focused on documents in English and covering characteristics of US-healthcare data providers.

As part of the IberLEF 2019 initiative we organize the first community challenge task specifically devoted to the anonymization of medical documents in Spanish, called the MEDDOCAN (Medical Document Anonymization) task.

In order to carry out these tasks we have prepared a synthetic corpus of 1000 clinical case studies. This corpus was selected manually by a practicing physician and augmented with PHI information from
discharge summaries and medical genetics clinical records.

The MEDDOCAN task will be structured into two sub-tasks:

NER offset and entity type classification.
Sensitive span detection.

## MEDDOCAN Corpus

The corpus was created by Montserrat Marimon et al. “Automatic De-identification of Medical Texts in Spanish: the MEDDOCAN Track, Corpus, Guidelines, Methods and Evaluation of Results.” In: IberLEF@ SEPLN. 2019, pp. 618–638.

For this task, They have prepared a synthetic corpus of clinical cases enriched with PHI expressions, named the MEDDOCAN corpus. This MEDDOCAN corpus of 1,000 clinical case studies was selected manually by a practicing physician and augmented with PHI phrases by health documentalists, adding PHI information from discharge summaries and medical genetics clinical records. See an example of MEDDOCAN annotation visualized using the BRAT annotation interface in *figure 1*.  

![Figure 1: An example of MEDDOCAN annotation visualized using the BRAT annotation interface.](https://temu.bsc.es/meddocan/wp-content/uploads/2019/03/image-1-1024x922.png)

More detailed information can be see [Description of the corpus](https://temu.bsc.es/meddocan/index.php/description-of-the-corpus/).
