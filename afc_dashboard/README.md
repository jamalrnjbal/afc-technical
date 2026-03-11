# AFC Fuhrpark-Dashboard

Interaktives Schadensanalyse-Dashboard für Auto Fleet Control.

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Diesen Ordner als GitHub-Repository hochladen
2. Auf https://share.streamlit.io einloggen
3. „New app" → Repository auswählen → `app.py` → Deploy
4. Fertig — öffentliche URL in ~2 Minuten

## Inhalt

- **Q1** · Kostenverlauf mit 3-Monats-Durchschnitt
- **Q2** · Schadensarten & Kostenvergleich zum Portfolio
- **Q3** · Reparaturdauer & offene Schäden
- **Q4** · Vermeidbare Schäden (Eigen-/Teilschuld)
- **Prognose** · Poisson GLM mit 80% Konfidenzband, Apr–Dez 2026
