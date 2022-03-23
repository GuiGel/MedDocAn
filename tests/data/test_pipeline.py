from meddocan.language.pipeline import meddocan_pipeline

if __name__ == "__main__":
    nlp = meddocan_pipeline()
    texts = [
        # "FerrándezCorreo",
        # "Remitido por: Dra. Lucrecia Sánchez-Rubio FerrándezCorreo electrónico: lsanchez@riojasalud.es",
        # "NºColMartínez",
        # "Médico: Gastón Demaría MartínezNºCol: 28 28 98702.",
        # "Médico: Gastón Demaría MartínezNºColumn: 28 28 98702.",
        "NºCol FerrándezCorreo",  # Si 2 mots sont dans le même document le test echoue...
    ]
    for doc in nlp.pipe(texts):
        print("=" * 100)
        for t in doc:
            print(t)
    print("=" * 100)
